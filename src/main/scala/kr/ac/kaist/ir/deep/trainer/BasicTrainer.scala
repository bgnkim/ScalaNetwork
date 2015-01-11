package kr.ac.kaist.ir.deep.trainer

import kr.ac.kaist.ir.deep.function._
import kr.ac.kaist.ir.deep.network.Network

import scala.annotation.tailrec

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
   * Tail Recursive : Train each batch
   * @param iter indicates current iteration
   * @param prevloss indicates previous loss
   * @param patience indicates current patience
   * @param isAutoEncoder is a flag for autoencoder.
   * @return Total Loss when train is finished
   */
  @tailrec
  protected final def trainBatch(iter: Int = 0,
                                 prevloss: Double = Double.MaxValue,
                                 patience: Int = stops.patience,
                                 isAutoEncoder: Boolean = false): Scalar = {
    trainingSet(param.miniBatch) foreach {
      pair ⇒ {
        val out = corrupt(pair._1) >>: net
        val err: ScalarMatrix = error.derivative(pair._2, out) / param.miniBatch.toDouble
        net ! err
      }
    }
    net.W -= net.dW

    var nPatience = patience

    val nLoss = if ((iter + 1) % stops.validationFreq == 0) {
      logger.debug(s"ITERATION $iter : W = ${net.W map (_.mkString) mkString " | "}")
      val train = validationError(isAutoEncoder)
      val weight = algorithm loss net.W
      if (train + weight < prevloss * stops.improveThreshold) {
        nPatience = Math.max(patience, iter * stops.patienceStep)
        bestIter = iter
        saveParams()
        logger.info(f"Iteration $iter%6d, Validation = $train%.5f, WeightLoss = $weight%.5f")
        train + weight
      } else {
        prevloss
      }
    } else {
      prevloss
    }

    if (iter < stops.maxIter && nPatience > iter && nLoss > stops.lossThreshold) {
      trainBatch(iter + 1, nLoss, nPatience, isAutoEncoder)
    } else {
      logger.info(f"Finished $iter%6d, Error = $nLoss%.5f")
      nLoss
    }
  }
}
