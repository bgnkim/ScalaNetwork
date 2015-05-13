package kr.ac.kaist.ir.deep.train

import kr.ac.kaist.ir.deep.fn.{WeightSeqOp, WeightUpdater}
import kr.ac.kaist.ir.deep.network.Network
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent._
import scala.concurrent.duration._
import scala.reflect.ClassTag

/**
 * __Trainer__ : Stochastic-Style, Multi-Threaded using Spark.
 *
 * @note This is not a implementation using DistBelief Paper.
 *       This is between [[DistBeliefTrainStyle]](DBTS) and [[SingleThreadTrainStyle]](STTS).
 *       The major difference is whether "updating" is asynchronous(DBTS) or not(MTTS).
 *
 * @param net __Network__ to be trained
 * @param algorithm Weight __update algorithm__ to be applied
 * @param make __Input Operation__ that supervises how to manipulate input as matrices.
 *             This also controls how to compute actual network. (default: [[VectorType]])
 * @param param __Training criteria__ (default: [[SimpleTrainingCriteria]])
 */
class MultiThreadTrainStyle[IN: ClassTag, OUT: ClassTag](override val net: Network,
                                                         override val algorithm: WeightUpdater,
                                                         @transient val sc: SparkContext,
                                                         override val make: ManipulationType[IN, OUT] = new VectorType(),
                                                         override val param: DistBeliefCriteria = DistBeliefCriteria())
  extends TrainStyle[IN, OUT] {
  /** Accumulator variable for networks */
  protected val accNet = sc.accumulator(net.dW)(WeightAccumulator)
  /** Accumulator variable for counter */
  protected val accCount = sc.accumulator(0)
  /** Training set */
  protected var trainingSet: RDD[Pair] = null
  /** Test Set */
  protected var testSet: RDD[Pair] = null

  /**
   * Unpersist all
   */
  def unpersist(): Unit = {
    if (trainingSet != null)
      trainingSet.unpersist()
    if (testSet != null)
      testSet.unpersist()
  }

  /**
   * Fetch weights
   *
   * @param iter current iteration
   */
  override def fetch(iter: Int): Unit = {
    accNet.value.map(_ := 0f)
    accCount.setValue(0)
  }

  /**
   * Send update of weights
   *
   * @param iter current iteration
   */
  override def update(iter: Int): Unit = {
    val dWUpdate = accNet.value
    val cnt = accCount.value.toFloat
    if (cnt > 0) {
      dWUpdate :/= cnt
      net.W -= dWUpdate
    } else {
      logger.warn(s"Epoch $iter trained with 0 instances. Please check.")
    }
  }

  /**
   * Do mini-batch
   */
  override def batch(): Unit =
    if (param.miniBatchFraction > 0) {
      val set = trainingSet.sample(withReplacement = true, fraction = param.miniBatchFraction)
      set.foreachPartition(partFunction)
      set.unpersist(blocking = false)
    } else {
      trainingSet.foreachPartition(partFunction)
    }

  private final def partFunction = {
    val netCopy = net.copy

    (part: Iterator[(IN, OUT)]) ⇒ {
      var count = 0
      val f = future {
        while (part.hasNext) {
          val pair = part.next()
          count += 1

          make.roundTrip(netCopy, make corrupted pair._1, pair._2)
        }

        accCount += count
        accNet += netCopy.dW
      }

      AsyncAwait.ready(f, 1.second)
    }
  }

  /**
   * Set training instances
   * @param set Sequence of training set
   */
  override def setPositiveTrainingReference(set: Seq[(IN, OUT)]): Unit = {
    val rdd =
      if (param.repartitionOnStart) sc.parallelize(set, param.numCores)
      else sc.parallelize(set)
    trainingSet = rdd.setName("Positives").persist(param.storageLevel)
    validationEpoch = if (param.miniBatchFraction > 0) Math.round(1.0f / param.miniBatchFraction) else 1
  }

  /**
   * Set training instances
   * @param set RDD of training set
   */
  override def setPositiveTrainingReference(set: RDD[(IN, OUT)]): Unit = {
    val rdd =
      if (param.repartitionOnStart) set.repartition(param.numCores).persist(param.storageLevel)
      else set
    trainingSet = rdd.setName(set.name + " (Positives)")
    validationEpoch = if (param.miniBatchFraction > 0) Math.round(1.0f / param.miniBatchFraction) else 1
  }

  /**
   * Set testing instances
   * @param set Sequence of testing set
   */
  override def setTestReference(set: Seq[(IN, OUT)]): Unit = {
    val rdd =
      if (param.repartitionOnStart) sc.parallelize(set, param.numCores)
      else sc.parallelize(set)
    testSet = rdd.setName("Validation").persist(param.storageLevel)
  }

  /**
   * Set testing instances
   * @param set RDD of testing set
   */
  override def setTestReference(set: RDD[(IN, OUT)]): Unit = {
    val rdd =
      if (param.repartitionOnStart) set.repartition(param.numCores).persist(param.storageLevel)
      else set
    testSet = rdd.setName(set.name + " (Validation)")
  }

  /**
   * Iterate over given number of test instances
   * @param n number of random sampled instances
   * @param fn iteratee function
   */
  override def foreachTestSet(n: Int)(fn: ((IN, OUT)) ⇒ Unit): Unit = {
    var seq = testSet.takeSample(withReplacement = true, num = n)
    while (seq.nonEmpty) {
      fn(seq.head)
      seq = seq.tail
    }
  }

  /**
   * Calculate validation error
   *
   * @return validation error
   */
  def validationError() = {
    val loss = sc.accumulator(0.0f)
    val count = sc.accumulator(0)
    val lossOf = make.lossOf(net) _
    testSet.foreachPartition {
      iter ⇒
        val f = future {
          var sum = 0.0f
          var c = 0
          while (iter.hasNext) {
            sum += lossOf(iter.next())
            c += 1
          }
          loss += sum
          count += c
        }

        AsyncAwait.ready(f, 1.second)
    }

    loss.value / count.value.toFloat
  }
}