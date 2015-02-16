package kr.ac.kaist.ir.deep.train

import kr.ac.kaist.ir.deep.fn._
import kr.ac.kaist.ir.deep.network._
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
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
class DistBeliefTrainStyle[IN, OUT](protected[train] override val net: Network,
                                    protected[train] override val algorithm: WeightUpdater,
                                    @transient protected val sc: SparkContext,
                                    protected[train] override val make: ManipulationType[IN, OUT] = new VectorType(),
                                    protected[train] override val param: DistBeliefCriteria = DistBeliefCriteria())
  extends TrainStyle[IN, OUT] {
  /** Spark distributed networks */
  @transient protected val networks = {
    val pairs = net copy param.numCores
    sc.makeRDD(pairs.indices map { id ⇒ id → pairs(id)}, param.numCores)
      .persist(StorageLevel.MEMORY_ONLY).cache()
  }

  /** Flag for fetch : Is fetching? */
  @transient protected var fetchFlag: Future[Unit] = null
  /** Flag for update : Is updating? */
  @transient protected var updateFlag: Future[Unit] = null

  /**
   * Fetch weights
   *
   * @param iter current iteration
   */
  override protected[train] def fetch(iter: Int): Unit =
    if (iter % param.fetchStep == 0) {
      if (fetchFlag != null && !fetchFlag.isCompleted) {
        logger warn s"Fetch command arrived before previous fetch is done. Need more steps between fetch commands!"
      }

      fetchFlag =
        future {
          val weights = sc.broadcast(net.W)
          networks foreach (_._2.W := weights.value)
          weights.destroy()
        }
    }

  /**
   * Send update of weights
   *
   * @param iter current iteration
   */
  override protected[train] def update(iter: Int): Unit =
    if (iter % param.updateStep == 0) {
      if (updateFlag != null && !updateFlag.isCompleted) {
        logger warn s"Update command arrived before previous update is done. Need more steps between update commands!"
      }

      updateFlag =
        future {
          val dWUpdate = networks.aggregate(IndexedSeq[ScalarMatrix]())({
            (seq, pair) ⇒
              val copiedNet = pair._2
              val out = copiedNet.dW copy_+ seq
              copiedNet.dW := 0.0
              out
          }, {
            case (dW1, dW2) if dW2.isEmpty ⇒ dW1
            case (dW1, dW2) if dW1.isEmpty ⇒ dW2
            case (dW1, dW2) ⇒
              dW1 :+= dW2
          })

          dWUpdate :/= (param.numCores * param.miniBatch).toDouble
          net.W -= dWUpdate
        }
    }

  /**
   * Indicates whether the asynchrononus update is finished or not.
   *
   * @return future object of update
   */
  override protected[train] def isUpdateFinished: Future[_] = updateFlag

  /**
   * Do mini-batch
   */
  override protected[train] def batch(): Unit = {
    val sets = sc.broadcast(trainingSet(param.miniBatch * param.numCores)
      .sliding(param.miniBatch, param.miniBatch).toSeq)

    val x = networks.foreachPartitionAsync {
      _ foreach {
        rddPair ⇒
          val copiedNet = rddPair._2
          val seq = sets.value(rddPair._1).map {
            pair ⇒ (make corrupted pair._1) → pair._2
          }
          make roundTrip(copiedNet, seq)
      }
    }

    x.onComplete {
      _ ⇒ sets.destroy()
    }

    try {
      Await.ready(x, 30.seconds)
    } catch {
      case _: Throwable ⇒
    }
  }
}
