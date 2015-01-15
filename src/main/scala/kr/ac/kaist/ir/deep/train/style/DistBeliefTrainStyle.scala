package kr.ac.kaist.ir.deep.train.style

import kr.ac.kaist.ir.deep.function._
import kr.ac.kaist.ir.deep.network.Network
import kr.ac.kaist.ir.deep.train._
import kr.ac.kaist.ir.deep.train.op.InputOp
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent._

/**
 * Train Style : Semi-DistBelief Style, Spark-based.
 *
 * Unlike with DistBelief, this trainer do updates and fetch by "master" not the "workers".
 *
 * @param net to be trained
 * @param algorithm to be applied
 * @param sc is a spark context that network will be distributed
 * @param param of training criteria (default: [[kr.ac.kaist.ir.deep.train.DistBeliefCriteria]])
 */
class DistBeliefTrainStyle[IN](protected[train] override val net: Network,
                               protected[train] override val algorithm: WeightUpdater,
                               @transient protected val sc: SparkContext,
                               protected[train] override val param: DistBeliefCriteria = DistBeliefCriteria())
  extends TrainStyle[IN] {
  /** Spark distributed networks */
  @transient protected val networks: RDD[Network] = sc.makeRDD(net copy param.numCores).persist(StorageLevel.MEMORY_ONLY).cache()
  /** Flags */
  @transient protected var fetchFlag: Boolean = false
  @transient protected var updateFlag: Boolean = false

  /**
   * Fetch weights 
   * @param iter is current iteration
   */
  override protected[train] def fetch(iter: Int): Unit =
    if (iter % param.fetchStep == 0 && !fetchFlag) {
      fetchFlag = true
      future {
        val weights = sc.broadcast(net.W)
        networks foreach (_.W := weights.value)
        weights.destroy()
      } onComplete {
        _ ⇒ fetchFlag = false
      }
    }

  /**
   * Send update of weights
   * @param iter is current iteration
   */
  override protected[train] def update(iter: Int): Unit =
    if (iter % param.updateStep == 0 && !updateFlag) {
      updateFlag = true
      future {
        val dWUpdate = networks.aggregate(Seq[ScalarMatrix]())({
          (seq, copiedNet) ⇒
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
   * Do mini-batch
   * @param op : set of input operations
   */
  override protected[train] def batch(op: InputOp[IN]): Unit =
    networks foreach {
      copiedNet ⇒
        trainingSet(param.miniBatch) map {
          pair ⇒ op roundTrip(copiedNet, op corrupted pair._1, pair._2)
        }
    }
}
