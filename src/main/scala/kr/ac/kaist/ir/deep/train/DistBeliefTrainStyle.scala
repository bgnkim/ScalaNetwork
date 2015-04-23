package kr.ac.kaist.ir.deep.train

import kr.ac.kaist.ir.deep.fn._
import kr.ac.kaist.ir.deep.network._
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast

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
class DistBeliefTrainStyle[IN: ClassTag, OUT: ClassTag](net: Network,
                                                        algorithm: WeightUpdater,
                                                        @transient sc: SparkContext,
                                                        make: ManipulationType[IN, OUT] = new VectorType(),
                                                        param: DistBeliefCriteria = DistBeliefCriteria())
  extends MultiThreadTrainStyle[IN, OUT](net, algorithm, sc, make, param) {
  /** Flag for batch : Is Batch remaining? */
  @transient protected var batchFlag = ArrayBuffer[Future[Unit]]()
  /** Flag for fetch : Is fetching? */
  @transient protected var fetchFlag: Future[Unit] = null
  /** Flag for update : Is updating? */
  @transient protected var updateFlag: Future[Unit] = null
  /** Spark distributed networks */
  protected var bcNet: Broadcast[Network] = _

  /**
   * Fetch weights
   *
   * @param iter current iteration
   */
  override def fetch(iter: Int): Unit =
    if (iter % param.fetchStep == 0) {
      if (fetchFlag != null && !fetchFlag.isCompleted) {
        logger warn "Fetch command arrived before previous fetch is done. Need more steps between fetch commands!"
      }

      fetchFlag =
        future {
          val oldNet = bcNet
          bcNet = sc.broadcast(net.copy)

          // Because DistBelief submit fetching job after n_fetch steps,
          // submit this fetch after already submitted jobs are done.
          // This does not block others because batch can be submitted anyway, 
          // and that batch does not affect this thread. 
          stopUntilBatchFinished()

          future {
            Thread.sleep(param.submitInterval.toMillis * param.fetchStep)
            oldNet.destroy()
          }
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
        logger warn "Update command arrived before previous update is done. Need more steps between update commands!"
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
          val count = accCount.value
          accCount.setValue(accCount.zero)

          dWUpdate :/= count.toFloat
          net.W -= dWUpdate
        }
    }

  /**
   * Non-blocking pending, until all assigned batches are finished
   */
  override def stopUntilBatchFinished(): Unit = {
    AsyncAwait.readyAll(param.submitInterval, batchFlag: _*)
    batchFlag = batchFlag.filterNot(_.isCompleted)
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
    val rddSet = trainingSet.sample(withReplacement = true, fraction = trainingFraction).repartition(param.numCores)
    val trainPair = if(negOutUniverse != null){
      negPartitioner.refreshRandom()
      negOutUniverse.sample(withReplacement = true, fraction = negFraction)
        .partitionBy(negPartitioner)
        .zipPartitions(rddSet) {
        (itNeg, itPair) ⇒
          itPair.map{
            pair ⇒
              val seq = ArrayBuffer[OUT]()
              seq.sizeHint(param.negSamplingRatio)

              while (seq.size < param.negSamplingRatio && itNeg.hasNext) {
                seq += itNeg.next()._2
              }

              (pair._1, pair._2, seq)
          }
      }
    }else rddSet.map(p ⇒ (p._1, p._2, Seq.empty[OUT]))

    val x = trainPair.foreachPartitionAsync {
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

    batchFlag += x

    x.onComplete {
      _ ⇒ rddSet.unpersist()
    }

    try {
      Await.ready(x, param.submitInterval)
    } catch {
      case _: Throwable ⇒
    }
  }
}