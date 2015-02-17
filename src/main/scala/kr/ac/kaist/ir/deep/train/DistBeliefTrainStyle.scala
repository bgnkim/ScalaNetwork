package kr.ac.kaist.ir.deep.train

import kr.ac.kaist.ir.deep.fn._
import kr.ac.kaist.ir.deep.network._
import org.apache.spark.SparkContext._
import org.apache.spark.{AccumulatorParam, SparkContext}

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
  /** Accumulator variable for networks */
  protected val accNet = sc.accumulator(net.dW)(WeightAccumulator)
  private val zeros = net.dW.map(_.copy)
  /** Spark distributed networks */
  protected var bcNet = sc.broadcast(net.copy)
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
  override protected[train] def update(iter: Int): Unit =
    if (iter % param.updateStep == 0) {
      if (updateFlag != null && !updateFlag.isCompleted) {
        logger warn s"Update command arrived before previous update is done. Need more steps between update commands!"
      }

      updateFlag =
        future {
          val dWUpdate = accNet.value
          accNet.setValue(zeros)

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
    val size = param.numCores * param.miniBatch
    val rddSet = sc.parallelize(trainingSet(size), param.numCores)

    val x = rddSet.foreachPartitionAsync {
      part ⇒
        val netCopy = bcNet.value.copy
        while (part.hasNext) {
          val pair = part.next()
          make.roundTrip(netCopy, make corrupted pair._1, pair._2)
        }
        accNet += netCopy.dW 
    }

    try {
      Await.ready(x, (size * 5).seconds)
    } catch {
      case _: Throwable ⇒
    }
  }
}

object WeightAccumulator extends AccumulatorParam[IndexedSeq[ScalarMatrix]] {
  override def addInPlace(r1: IndexedSeq[ScalarMatrix], r2: IndexedSeq[ScalarMatrix]): IndexedSeq[ScalarMatrix] = {
    r1 :+= r2
  }

  override def zero(initialValue: IndexedSeq[ScalarMatrix]): IndexedSeq[ScalarMatrix] = initialValue.map(_.copy)
}