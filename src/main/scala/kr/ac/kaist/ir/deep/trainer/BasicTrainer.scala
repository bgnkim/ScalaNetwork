package kr.ac.kaist.ir.deep.trainer

import kr.ac.kaist.ir.deep.function._
import kr.ac.kaist.ir.deep.network.Network

/**
 * Trainer : Stochastic-Style
 * @param net to be trained
 * @param algorithm to be applied
 * @param error to compute loss (default: [[kr.ac.kaist.ir.deep.function.SquaredErr]])
 * @param corrupt to corrupt input (default: [[kr.ac.kaist.ir.deep.trainer.NoCorruption]])
 * @param param of training criteria (default: [[kr.ac.kaist.ir.deep.trainer.SimpleTrainingCriteria]])
 * @param stops of stopping criteria (default: [[kr.ac.kaist.ir.deep.trainer.StoppingCriteria]])
 */
class BasicTrainer(protected override val net: Network,
                   protected override val algorithm: WeightUpdater,
                   protected override val error: Objective = SquaredErr,
                   protected override val corrupt: Corruption = NoCorruption,
                   protected override val param: TrainingCriteria = SimpleTrainingCriteria(),
                   protected override val stops: StoppingCriteria = StoppingCriteria())
  extends Trainer {

  /**
   * Fetch weights 
   * @param iter is current iteration
   */
  override protected def fetch(iter: Int): Unit = {}

  /**
   * Send update of weights  
   * @param iter is current iteration
   */
  override protected def update(iter: Int): Unit = {
    net.W -= net.dW
  }

  /**
   * Do mini-batch
   */
  override protected def batch(): Unit =
    trainingSet(param.miniBatch) foreach {
      pair ⇒ {
        val out = corrupt(pair._1) >>: net
        val err: ScalarMatrix = error.derivative(pair._2, out) / param.miniBatch.toDouble
        net ! err
      }
    }
}
