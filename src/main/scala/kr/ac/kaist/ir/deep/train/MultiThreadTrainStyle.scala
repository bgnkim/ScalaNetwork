package kr.ac.kaist.ir.deep.train

import kr.ac.kaist.ir.deep.fn.{WeightSeqOp, WeightUpdater}
import kr.ac.kaist.ir.deep.network.Network
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer
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
  /** Spark distributed networks */
  protected var bcNet = sc.broadcast(net.copy)
  /** Training set */
  protected var trainingSet: RDD[Pair] = null
  /** Fraction of mini-batch */
  protected var trainingFraction: Float = 0.0f
  /** Negative Sampler */
  protected var negOutUniverse: RDD[(Long, OUT)] = null
  /** Partitioner for negative samples */
  protected var negPartitioner: RandomEqualPartitioner = _
  /** Fraction of negative samples */
  protected var negFraction: Float = 0.0f
  /** Test Set */
  protected var testSet: RDD[Pair] = null
  /** Size of test set */
  protected var testSize: Float = 0.0f

  /**
   * Unpersist all
   */
  def unpersist(): Unit = {
    if (trainingSet != null)
      trainingSet.unpersist()
    if (negOutUniverse != null)
      negOutUniverse.unpersist()
    if (testSet != null)
      testSet.unpersist()
  }

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
    val dWUpdate = accNet.value
    val cnt = accCount.value
    if (cnt > 0) {
      accNet.setValue(accNet.zero)
      accCount.setValue(accCount.zero)
      bcNet.unpersist(blocking = false)

      dWUpdate :/= cnt.toFloat
      net.W -= dWUpdate

      bcNet = sc.broadcast(net.copy)
    } else {
      logger.warn(s"This iteration trained with 0 instances. Please check.")
    }
  }

  /**
   * Do mini-batch
   */
  override def batch(): Unit = {
    val rddSet = trainingSet.sample(withReplacement = true, fraction = trainingFraction)
    val trainPair = if (negOutUniverse != null) {
      negPartitioner.refreshRandom()
      negOutUniverse.sample(withReplacement = true, fraction = negFraction)
        .partitionBy(negPartitioner)
        .zipPartitions(rddSet) {
        (itNeg, itPair) ⇒
          itPair.map {
            pair ⇒
              val seq = ArrayBuffer[OUT]()
              seq.sizeHint(param.negSamplingRatio)

              while (seq.size < param.negSamplingRatio && itNeg.hasNext) {
                seq += itNeg.next()._2
              }

              (pair._1, pair._2, seq)
          }
      }
    } else rddSet.map(p ⇒ (p._1, p._2, Seq.empty[OUT]))

    trainPair.foreachPartition {
      part ⇒
        val netCopy = bcNet.value.copy
        var count = 0

        val f = Future.traverse(part) {
          pair ⇒
            count += 1

            future {
              make.roundTrip(netCopy, make corrupted pair._1, pair._2)

              var samples = pair._3
              while (samples.nonEmpty) {
                val neg = samples.head
                samples = samples.tail

                if (make.different(neg, pair._2))
                  make.roundTrip(netCopy, make corrupted pair._1, neg, isPositive = false)
              }
            }
        }

        AsyncAwait.ready(f, 1.second)
        accCount += count
        accNet += netCopy.dW
    }

    rddSet.unpersist()
  }

  /**
   * Set training instances
   * @param set Sequence of training set
   */
  override def setPositiveTrainingReference(set: Seq[(IN, OUT)]): Unit = {
    trainingSet = sc.parallelize(set, param.numCores)
      .setName("Positives").persist(param.storageLevel)
    trainingFraction = param.miniBatch / set.size.toFloat
    validationEpoch = set.size / param.miniBatch
  }

  /**
   * Set training instances
   * @param set RDD of training set
   */
  override def setPositiveTrainingReference(set: RDD[(IN, OUT)]): Unit = {
    trainingSet = set.repartition(param.numCores)
      .setName(set.name + " (Positives)").persist(param.storageLevel)
    val count = trainingSet.count()
    trainingFraction = param.miniBatch / count.toFloat
    validationEpoch = (count / param.miniBatch).toInt
  }

  /**
   * Set negative sampling method.
   * @param set all training outputs that will be used for negative training
   */
  override def setNegativeTrainingReference(set: Seq[OUT]): Unit = {
    negOutUniverse = sc.parallelize(set, param.numCores).zipWithUniqueId().map(_.swap)
      .setName("Negatives").persist(param.storageLevel)
    val size = set.size
    negFraction = (param.miniBatch * param.negSamplingRatio * 2) / size.toFloat
    negPartitioner = new RandomEqualPartitioner(param.numCores)
  }

  /**
   * Set negative sampling method.
   * @param set all training outputs that will be used for negative training
   */
  override def setNegativeTrainingReference(set: RDD[OUT]): Unit = {
    negOutUniverse = set.repartition(param.numCores).zipWithUniqueId().map(_.swap)
      .setName(set.name + " (Negatives)").persist(param.storageLevel)
    val size = set.count()
    negFraction = (param.miniBatch * param.negSamplingRatio * 2) / size.toFloat
    negPartitioner = new RandomEqualPartitioner(param.numCores)
  }

  /**
   * Set testing instances
   * @param set Sequence of testing set
   */
  override def setTestReference(set: Seq[(IN, OUT)]): Unit = {
    testSet = sc.parallelize(set, param.numCores)
      .setName("Validation").persist(param.storageLevel)
    testSize = set.size.toFloat
  }

  /**
   * Set testing instances
   * @param set RDD of testing set
   */
  override def setTestReference(set: RDD[(IN, OUT)]): Unit = {
    testSet = set.repartition(param.numCores)
      .setName(set.name + " (Validation)").persist(param.storageLevel)
    testSize = testSet.count().toFloat
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
    val lossOf = make.lossOf(net) _
    testSet.foreachPartition {
      iter ⇒
        var sum = 0.0f
        while (iter.hasNext) {
          sum += lossOf(iter.next()) / testSize
        }
        loss += sum
    }

    loss.value
  }
}