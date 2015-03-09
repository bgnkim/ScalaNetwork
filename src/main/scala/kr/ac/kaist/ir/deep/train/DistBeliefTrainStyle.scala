package kr.ac.kaist.ir.deep.train

import java.util.concurrent.ThreadLocalRandom

import kr.ac.kaist.ir.deep.fn._
import kr.ac.kaist.ir.deep.network._
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import scala.collection.mutable.ArrayBuffer
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent._
import scala.concurrent.duration._
import scala.reflect._

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
class DistBeliefTrainStyle[IN:ClassTag, OUT:ClassTag](override val net: Network,
                                                      override val algorithm: WeightUpdater,
                                                      @transient val sc: SparkContext,
                                                      override val make: ManipulationType[IN, OUT] = new VectorType(),
                                                      override val param: DistBeliefCriteria = DistBeliefCriteria())
  extends TrainStyle[IN, OUT] {
  /** Accumulator variable for networks */
  protected val accNet = sc.accumulator(net.dW)(WeightAccumulator)
  /** Flag for batch : Is Batch remaining? */
  @transient protected val batchFlag = ArrayBuffer[Future[Unit]]()
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
  private var negOutUniverse: RDD[OUT] = null
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
          // Because DistBelief submit fetching job after n_fetch steps,
          // submit this fetch after already submitted jobs are done.
          // This does not block others because batch can be submitted anyway, 
          // and that batch does not affect this thread. 
          stopUntilBatchFinished()
          
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

      // Repartition negative samples.
      future {
        val old = negOutUniverse
        negOutUniverse = old.repartition(param.numCores)
        old.unpersist()
      }

      updateFlag =
        future {
          // Because DistBelief submit updating job after n_update steps,
          // Submit this update after already submitted jobs are done.
          // This does not block others because batch can be submitted anyway,
          // and that batch does not affect this thread.
          stopUntilBatchFinished()

          val dWUpdate = accNet.value
          accNet.setValue(accNet.zero)

          dWUpdate :/= (param.numCores * param.miniBatch).toFloat
          net.W -= dWUpdate
        }
    }

  /**
   * Non-blocking pending, until all assigned batches are finished
   */
  override def stopUntilBatchFinished(): Unit =
    AsyncAwait.readyAll(param.submitInterval, batchFlag: _*)

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
    val rddSet = trainingSet.sample(withReplacement = true, fraction = trainingFraction)
    val trainPair = if(negOutUniverse != null){
      rddSet.zipPartitions(negOutUniverse){
        val rand = ThreadLocalRandom.current()

        (itPair, itNeg) ⇒
          var (orig, copy) = itNeg.duplicate
          itPair.map{
            pair ⇒
              val seq = (1 to param.negSamplingRatio).map{
                _ ⇒
                  do{
                    copy.next()
                    if(!copy.hasNext) {
                      val (orig2, copy2) = orig.duplicate
                      orig = orig2
                      copy = copy2
                    }
                  }while(rand.nextFloat() < 0.3f)

                  copy.next()
              }.toSeq
              (pair._1, pair._2, seq)
          }
      }
    }else rddSet.map(p ⇒ (p._1, p._2, Seq.empty[OUT]))

    val x = future {
      trainPair.foreachPartition {
        val useNeg = param.negSamplingRatio > 0
        part ⇒
          val netCopy = bcNet.value.copy

          val f = part.map {
            pair ⇒
              future {
                make.roundTrip(netCopy, make corrupted pair._1, pair._2)

                var samples = pair._3
                while (samples.nonEmpty) {
                  val neg = samples.head
                  samples = samples.tail

                  if(make.different(neg, pair._2))
                    make.roundTrip(netCopy, make corrupted pair._1, neg, isPositive = false)
                }
              }
          }.toSeq

          AsyncAwait.readyAll(1.second, f: _*)
          accNet += netCopy.dW
      }
    }

    batchFlag += x

    x.onComplete {
      _ ⇒
        rddSet.unpersist()
        batchFlag -= x
    }

    try {
      Await.ready(x, param.submitInterval)
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
    trainingFraction = param.miniBatch / set.size.toFloat
  }

  /**
   * Set training instances
   * @param set RDD of training set
   */
  override def setPositiveTrainingReference(set: RDD[(IN, OUT)]): Unit = {
    trainingSet = set.repartition(param.numCores).persist(StorageLevel.DISK_ONLY_2)
    trainingFraction = param.miniBatch / set.count().toFloat
  }

  /**
   * Set negative sampling method.
   * @param set all training outputs that will be used for negative training
   */
  override def setNegativeTrainingReference(set: Seq[OUT]): Unit = {
    negOutUniverse = sc.parallelize(set, param.numCores).persist(StorageLevel.DISK_ONLY_2)
  }

  /**
   * Set negative sampling method.
   * @param set all training outputs that will be used for negative training
   */
  override def setNegativeTrainingReference(set: RDD[OUT]): Unit = {
    negOutUniverse = set.repartition(param.numCores).persist(StorageLevel.DISK_ONLY_2)
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