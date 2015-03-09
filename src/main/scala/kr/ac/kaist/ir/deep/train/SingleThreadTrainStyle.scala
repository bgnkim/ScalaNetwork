package kr.ac.kaist.ir.deep.train

import kr.ac.kaist.ir.deep.fn.{WeightSeqOp, WeightUpdater}
import kr.ac.kaist.ir.deep.network.Network
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import scala.collection.mutable.ArrayBuffer

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
  private var trainingSet: Int ⇒ Seq[Pair] = null
  /** Negative Sampler */
  private var negOutUniverse: Int ⇒ Seq[OUT] = null
  /** Test Set */
  private var testSet: Int ⇒ Seq[Pair] = null

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
    net.dW :/= param.miniBatch.toFloat
    net.W -= net.dW
  }

  /**
   * Do mini-batch
   */
  override def batch(): Unit = {
    var seq = trainingSet(param.miniBatch)
    if (param.negSamplingRatio == 0 || negOutUniverse == null) {
      while (seq.nonEmpty) {
        val pair = seq.head
        seq = seq.tail
        make roundTrip(net, make corrupted pair._1, pair._2)
      }
    } else {
      while (seq.nonEmpty) {
        val pair = seq.head
        seq = seq.tail
        make roundTrip(net, make corrupted pair._1, pair._2)

        var samples = negOutUniverse(param.negSamplingRatio)
        while (samples.nonEmpty) {
          val neg = samples.head
          samples = samples.tail
          make roundTrip(net, make corrupted pair._1, neg, isPositive = false)
        }
      }
    }
  }

  /**
   * Set training instances 
   * @param set Sequence of training set
   */
  override def setPositiveTrainingReference(set: Seq[(IN, OUT)]): Unit = {
    val index = () ⇒ Math.floor(Math.random() * set.size).toInt
    trainingSet = (n: Int) ⇒ {
      val array = ArrayBuffer[Pair]()
      array.sizeHint(n)

      var i = 0
      while (i < n) {
        array += set(index())
        i += 1
      }
      array
    }
  }

  /**
   * Set training instances
   * @param set RDD of training set
   */
  override def setPositiveTrainingReference(set: RDD[(IN, OUT)]): Unit = {
    trainingSet = (n: Int) ⇒ set.takeSample(withReplacement = true, num = n).toSeq
  }

  /**
   * Set negative sampling method.
   * @param set all training outputs that will be used for negative training
   */
  override def setNegativeTrainingReference(set: Seq[OUT]): Unit = {
    val index = () ⇒ Math.floor(Math.random() * set.size).toInt
    negOutUniverse = (n: Int) ⇒ {
      val array = ArrayBuffer[OUT]()
      array.sizeHint(n)

      var i = 0
      while (i < n) {
        array += set(index())
        i += 1
      }
      array
    }
  }

  /**
   * Set negative sampling method.
   * @param set all training outputs that will be used for negative training
   */
  override def setNegativeTrainingReference(set: RDD[OUT]): Unit = {
    negOutUniverse = (n: Int) ⇒ set.takeSample(withReplacement = true, num = n).toSeq
  }

  /**
   * Set testing instances 
   * @param set Sequence of testing set
   */
  override def setTestReference(set: Seq[(IN, OUT)]): Unit = {
    testSet = set.take
  }

  /**
   * Set testing instances
   * @param set RDD of testing set
   */
  override def setTestReference(set: RDD[(IN, OUT)]): Unit = {
    testSet = (n: Int) ⇒ set.takeSample(withReplacement = true, num = n).toSeq
  }

  /**
   * Calculate validation error
   *
   * @return validation error
   */
  def validationError() = {
    val size = param.validationSize
    val lossOf = make.lossOf(net) _
    var sum = 0.0f
    foreachTestSet(size) {
      item ⇒ sum += lossOf(item) / size
    }
    sum
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
