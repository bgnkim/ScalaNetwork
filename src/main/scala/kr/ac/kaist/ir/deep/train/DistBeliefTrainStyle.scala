package kr.ac.kaist.ir.deep.train

import kr.ac.kaist.ir.deep.fn._
import kr.ac.kaist.ir.deep.network.Network
import org.apache.spark.SparkContext
import org.apache.spark.storage.StorageLevel

import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent._

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
 * @param param __DistBelief-style__ Training criteria (default: [[kr.ac.kaist.ir.deep.train.DistBeliefCriteria]])
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
  @transient protected var fetchFlag: Boolean = false
  /** Flag for update : Is updating? */
  @transient protected var updateFlag: Boolean = false

  /**
   * Fetch weights
   *
   * @param iter current iteration
   */
  override protected[train] def fetch(iter: Int): Unit =
    if (iter % param.fetchStep == 0 && !fetchFlag) {
      fetchFlag = true
      future {
        val weights = sc.broadcast(net.W)
        networks foreach (_._2.W := weights.value)
        weights.destroy()
      } onComplete {
        _ ⇒ fetchFlag = false
      }
    }

  /**
   * Send update of weights
   *
   * @param iter current iteration
   */
  override protected[train] def update(iter: Int): Unit =
    if (iter % param.updateStep == 0 && !updateFlag) {
      updateFlag = true
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
      } onComplete {
        _ ⇒ updateFlag = false
      }
    }

  /**
   * Indicates whether the asynchrononus update is finished or not.
   *
   * @return boolean flag of update
   */
  override protected[train] def isUpdateFinished: Boolean = updateFlag

  /**
   * Do mini-batch
   */
  override protected[train] def batch(): Unit = {
    val sets = (0 until param.numCores) map {
      _ ⇒ sc.broadcast(trainingSet(param.miniBatch))
    }

    networks foreachPartition {
      _ foreach {
        rddPair ⇒
          val copiedNet = rddPair._2
          val seq = sets(rddPair._1).value.map {
            pair ⇒ (make corrupted pair._1) → pair._2
          }.seq
          make roundTrip(copiedNet, seq)
      }
    }

    sets map {
      _.destroy()
    }
  }
}
