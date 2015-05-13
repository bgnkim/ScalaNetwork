package kr.ac.kaist.ir.deep.train

import java.util.concurrent.ThreadLocalRandom

import kr.ac.kaist.ir.deep.fn.{Scalar, WeightSeqOp, WeightUpdater}
import kr.ac.kaist.ir.deep.network.Network
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

/**
 * __Trainer__ : Stochastic-Style, Single-Threaded
 *
 * @param net __Network__ to be trained
 * @param algorithm Weight __update algorithm__ to be applied
 * @param make __Input Operation__ that supervises how to manipulate input as matrices.
 *             This also controls how to compute actual network. (default: [[VectorType]])
 * @param param __Training criteria__ (default: [[SimpleTrainingCriteria]])
 */
class SingleThreadTrainStyle[IN, OUT](override val net: Network,
                                      override val algorithm: WeightUpdater,
                                      override val make: ManipulationType[IN, OUT] = new VectorType(),
                                      override val param: TrainingCriteria = SimpleTrainingCriteria())
  extends TrainStyle[IN, OUT] {

  /** Training set */
  private var trainingSet: Scalar ⇒ Seq[Pair] = null
  /** Test Set */
  private var testSet: Int ⇒ Seq[Pair] = null
  /** Test Set iterator */
  private var testSetMapper: (Pair ⇒ Unit) ⇒ Unit = null
  /** Test Set Context. Null if testset is a local seq */
  private var testSetSC: SparkContext = null
  /** Count */
  private var count = 0

  /**
   * Fetch weights
   *
   * @param iter current iteration
   */
  override def fetch(iter: Int): Unit = {}

  /**
   * Send update of weights
   *
   * @param iter current iteration
   */
  override def update(iter: Int): Unit = {
    net.dW :/= count.toFloat
    net.W -= net.dW
    count = 0
  }

  /**
   * Do mini-batch
   */
  override def batch(): Unit = {
    var seq = trainingSet(param.miniBatchFraction)
    while (seq.nonEmpty) {
      val pair = seq.head
      seq = seq.tail
      count += 1
      make roundTrip(net, make corrupted pair._1, pair._2)
    }
  }

  /**
   * Set training instances 
   * @param set Sequence of training set
   */
  override def setPositiveTrainingReference(set: Seq[(IN, OUT)]): Unit = {
    trainingSet = (x: Scalar) ⇒
      if (x > 0) {
        set.filter(_ ⇒ ThreadLocalRandom.current().nextFloat() < x)
      } else {
        set
      }
    validationEpoch = if (param.miniBatchFraction > 0) Math.round(1.0f / param.miniBatchFraction) else 1
  }

  /**
   * Set training instances
   * @param set RDD of training set
   */
  override def setPositiveTrainingReference(set: RDD[(IN, OUT)]): Unit = {
    trainingSet = (x: Scalar) ⇒
      if (x > 0) set.sample(withReplacement = true, fraction = x).collect().toSeq
      else set.collect()
    validationEpoch = if (param.miniBatchFraction > 0) Math.round(1.0f / param.miniBatchFraction) else 1
  }

  /**
   * Set testing instances 
   * @param set Sequence of testing set
   */
  override def setTestReference(set: Seq[(IN, OUT)]): Unit = {
    testSet = set.take
    testSetMapper = (mapper: Pair ⇒ Unit) ⇒ {
      var seq = set
      while (seq.nonEmpty) {
        mapper(seq.head)
        seq = seq.tail
      }
    }
    testSetSC = null
  }

  /**
   * Set testing instances
   * @param set RDD of testing set
   */
  override def setTestReference(set: RDD[(IN, OUT)]): Unit = {
    testSet = (n: Int) ⇒ set.takeSample(withReplacement = true, num = n).toSeq
    testSetMapper = (mapper: Pair ⇒ Unit) ⇒ {
      set.foreach(mapper)
    }
    testSetSC = set.context
  }

  /**
   * Calculate validation error
   *
   * @return validation error
   */
  def validationError() = {
    val lossOf = make.lossOf(net) _

    if (testSetSC == null) {
      // If it is from general "local" sequence
      var sum = 0.0f
      var count = 0
      testSetMapper {
        item ⇒
          sum += lossOf(item)
          count += 1
      }
      sum / count.toFloat
    } else {
      // If it is from RDD
      val sum = testSetSC.accumulator(0.0f)
      val count = testSetSC.accumulator(0)
      val bcLoss = testSetSC.broadcast(lossOf)
      testSetMapper {
        item ⇒
          sum += bcLoss.value(item)
          count += 1
      }
      bcLoss.destroy()
      sum.value / count.value.toFloat
    }
  }

  /**
   * Iterate over given number of test instances 
   * @param n number of random sampled instances
   * @param fn iteratee function
   */
  override def foreachTestSet(n: Int)(fn: ((IN, OUT)) ⇒ Unit): Unit = {
    var set = testSet(n)
    while (set.nonEmpty) {
      fn(set.head)
      set = set.tail
    }
  }
}
