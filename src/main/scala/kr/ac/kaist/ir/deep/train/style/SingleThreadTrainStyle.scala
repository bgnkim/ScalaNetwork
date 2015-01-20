package kr.ac.kaist.ir.deep.train.style

import kr.ac.kaist.ir.deep.fn.WeightSeqOp
import kr.ac.kaist.ir.deep.fn.alg.WeightUpdater
import kr.ac.kaist.ir.deep.network.Network
import kr.ac.kaist.ir.deep.train._
import kr.ac.kaist.ir.deep.train.op.InputOp

/**
 * Trainer : Stochastic-Style, Single-Threaded
 *
 * @param net to be trained
 * @param algorithm to be applied
 * @param param of training criteria (default: [[kr.ac.kaist.ir.deep.train.SimpleTrainingCriteria]])
 */
class SingleThreadTrainStyle[IN](protected[train] override val net: Network,
                                 protected[train] override val algorithm: WeightUpdater,
                                 protected[train] override val param: TrainingCriteria = SimpleTrainingCriteria())
  extends TrainStyle[IN] {

  /**
   * Fetch weights 
   * @param iter is current iteration
   */
  override protected[train] def fetch(iter: Int): Unit = {}

  /**
   * Send update of weights  
   * @param iter is current iteration
   */
  override protected[train] def update(iter: Int): Unit = {
    net.dW :/= param.miniBatch.toDouble
    net.W -= net.dW
  }

  /**
   * Do mini-batch
   * @param op : set of input operations
   */
  override protected[train] def batch(op: InputOp[IN]): Unit =
    trainingSet(param.miniBatch) foreach {
      pair ⇒ op roundTrip(net, op corrupted pair._1, pair._2)
    }
}
