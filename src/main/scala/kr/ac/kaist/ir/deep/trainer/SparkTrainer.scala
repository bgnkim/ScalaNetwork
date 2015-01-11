package kr.ac.kaist.ir.deep.trainer

import kr.ac.kaist.ir.deep.function._
import kr.ac.kaist.ir.deep.network.Network
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import scala.annotation.tailrec
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent._

/**
 * Trainer : Semi-DistBelief Style, Spark-based.
 *
 * Unlike with DistBelief, this trainer do updates and fetch by "master" not the "workers".
 *
 * @param net to be trained
 * @param algorithm to be applied
 * @param sc is a spark context that network will be distributed
 * @param error to compute loss (default: [[kr.ac.kaist.ir.deep.function.SquaredErr]])
 * @param corrupt to corrupt input (default: [[kr.ac.kaist.ir.deep.trainer.NoCorruption]])
 * @param param of training criteria (default: [[kr.ac.kaist.ir.deep.trainer.DistBeliefCriteria]])
 * @param stops of stopping criteria (default: [[kr.ac.kaist.ir.deep.trainer.StoppingCriteria]])
 */
class SparkTrainer(protected override val net: Network,
                   protected override val algorithm: WeightUpdater,
                   @transient private val sc: SparkContext,
                   protected override val error: Objective = SquaredErr,
                   protected override val corrupt: Corruption = NoCorruption,
                   protected override val param: DistBeliefCriteria = DistBeliefCriteria(),
                   protected override val stops: StoppingCriteria = StoppingCriteria())
  extends Trainer {
  /** Spark distributed networks */
  @transient protected val networks: RDD[Network] = sc.makeRDD(net copy param.numCores).persist(StorageLevel.MEMORY_ONLY).cache()
  /** Flags */
  @transient protected var fetchFlag: Boolean = false
  @transient protected var updateFlag: Boolean = false

  /**
   * Train using given RDD sequence. 
   * @param set to be used for training
   */
  def train(set: RDD[(ScalarMatrix, ScalarMatrix)]): Scalar = {
    set.cache()
    train(set.takeSample(true, _), set.top(_))
  }

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
    // if it is a fetch step, distribute network.
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

    // For each training set, do train.
    networks foreach {
      copiedNet ⇒
        trainingSet(param.miniBatch) map {
          pair ⇒ {
            val in = corrupt(pair._1)
            val out = in >>: copiedNet
            val err: ScalarMatrix = error.derivative(pair._2, out) / param.miniBatch.toDouble
            copiedNet ! err
          }
        }
    }

    // If this is update step, collect update.
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

        net.W -= dWUpdate
      } onComplete {
        _ ⇒ updateFlag = false
      }
    }

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

    if (iter < param.miniBatch * stops.maxIter && nPatience > iter && nLoss > stops.lossThreshold) {
      trainBatch(iter + 1, nLoss, nPatience, isAutoEncoder)
    } else {
      logger.info(f"Finished $iter%6d, Error = $nLoss%.5f")
      nLoss
    }
  }
}
