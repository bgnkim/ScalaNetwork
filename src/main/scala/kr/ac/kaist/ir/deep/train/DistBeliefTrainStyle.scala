package kr.ac.kaist.ir.deep.train

import kr.ac.kaist.ir.deep.fn._
import kr.ac.kaist.ir.deep.network._
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent._
import scala.concurrent.duration._

/**
 * __Train Style__ : Semi-DistBelief Style, Spark-based.
 *
 * @note Unlike with DistBelief, this trainer do updates and fetch by '''master''' not the '''workers'''.
 *
 * @param net __Network__ to be trained
 * @param algorithm Weight __update algorithm__ to be applied
 * @param sc A __spark context__ that network will be distributed
 * @param make __Input Operation__ that supervises how to manipulate input as matrices.
 *             This also controls how to compute actual network. (default: [[VectorType]])
 * @param param __DistBelief-style__ Training criteria (default: [[DistBeliefCriteria]])
 */
class DistBeliefTrainStyle[IN, OUT](override val net: Network,
                                    override val algorithm: WeightUpdater,
                                    @transient val sc: SparkContext,
                                    override val make: ManipulationType[IN, OUT] = new VectorType(),
                                    override val param: DistBeliefCriteria = DistBeliefCriteria())
  extends TrainStyle[IN, OUT] {
  /** Accumulator variable for networks */
  protected val accNet = sc.accumulator(net.dW)(WeightAccumulator)
  private val zeros = net.dW.map(_.copy)
  /** Spark distributed networks */
  protected var bcNet = sc.broadcast(net.copy)
  /** Flag for fetch : Is fetching? */
  @transient protected var fetchFlag: Future[Unit] = null
  /** Flag for update : Is updating? */
  @transient protected var updateFlag: Future[Unit] = null

  /** Training set */
  private var trainingSet: RDD[Pair] = null
  /** Fraction of mini-batch */
  private var trainingFraction: Float = 0.0f
  /** Negative Sampler */
  private var negativeSampler: Broadcast[Sampler] = sc.broadcast(null)
  /** Test Set */
  private var testSet: RDD[Pair] = null
  /** Size of test set */
  private var testSize: Float = 0.0f
  
  /**
   * Fetch weights
   *
   * @param iter current iteration
   */
  override def fetch(iter: Int): Unit =
    if (iter % param.fetchStep == 0) {
      if (fetchFlag != null && !fetchFlag.isCompleted) {
        logger warn s"Fetch command arrived before previous fetch is done. Need more steps between fetch commands!"
      }

      fetchFlag =
        future {
          val oldNet = bcNet
          bcNet = sc.broadcast(net.copy)
          oldNet.destroy()
        }
    }

  /**
   * Send update of weights
   *
   * @param iter current iteration
   */
  override def update(iter: Int): Unit =
    if (iter % param.updateStep == 0) {
      if (updateFlag != null && !updateFlag.isCompleted) {
        logger warn s"Update command arrived before previous update is done. Need more steps between update commands!"
      }

      updateFlag =
        future {
          val dWUpdate = accNet.value
          accNet.setValue(zeros)

          dWUpdate :/= (param.numCores * param.miniBatch).toFloat
          net.W -= dWUpdate
        }
    }

  /**
   * Indicates whether the asynchrononus update is finished or not.
   *
   * @return future object of update
   */
  override def isUpdateFinished: Future[_] = updateFlag

  /**
   * Do mini-batch
   */
  override def batch(): Unit = {
    val size = param.numCores * param.miniBatch
    val rddSet = trainingSet.sample(withReplacement = true, fraction = trainingFraction)

    val x = future {
      rddSet.foreachPartition {
        val sampler = negativeSampler.value
        val useNeg = sampler != null && param.negSamplingRatio > 0
        part ⇒
          val netCopy = bcNet.value.copy
          while (part.hasNext) {
            val pair = part.next()
            make.roundTrip(netCopy, make corrupted pair._1, pair._2)

            if (useNeg) {
              var samples = sampler(pair._1, param.negSamplingRatio)
              while (samples.nonEmpty) {
                val neg = samples.head
                samples = samples.tail

                make.roundTrip(netCopy, make corrupted pair._1, neg, isPositive = false)
              }
            }
          }
          accNet += netCopy.dW
      }
    }

    x.onComplete {
      _ ⇒ rddSet.unpersist()
    }

    try {
      Await.ready(x, (size * 5).seconds)
    } catch {
      case _: Throwable ⇒
    }
  }


  /**
   * Set training instances 
   * @param set Sequence of training set
   */
  override def setPositiveTrainingReference(set: Seq[(IN, OUT)]): Unit = {
    trainingSet = sc.parallelize(set, param.numCores).persist(StorageLevel.DISK_ONLY_2)
    trainingFraction = set.size.toFloat / param.miniBatch
  }

  /**
   * Set training instances
   * @param set RDD of training set
   */
  override def setPositiveTrainingReference(set: RDD[(IN, OUT)]): Unit = {
    trainingSet = set.repartition(param.numCores).persist(StorageLevel.DISK_ONLY_2)
    trainingFraction = set.count().toFloat / param.miniBatch
  }

  /**
   * Set negative sampling method.
   * @param set Sampler function
   */
  override def setNegativeSampler(set: Sampler): Unit = {
    negativeSampler.destroy()
    negativeSampler = sc.broadcast(set)
  }

  /**
   * Set testing instances 
   * @param set Sequence of testing set
   */
  override def setTestReference(set: Seq[(IN, OUT)]): Unit = {
    testSet = sc.parallelize(set, param.numCores).persist(StorageLevel.DISK_ONLY_2)
    testSize = set.size.toFloat
  }

  /**
   * Set testing instances
   * @param set RDD of testing set
   */
  override def setTestReference(set: RDD[(IN, OUT)]): Unit = {
    testSet = set.repartition(param.numCores).persist(StorageLevel.DISK_ONLY_2)
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