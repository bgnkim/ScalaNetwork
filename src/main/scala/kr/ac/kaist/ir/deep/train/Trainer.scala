package kr.ac.kaist.ir.deep.train

import kr.ac.kaist.ir.deep.fn._
import org.apache.log4j.Logger
import org.apache.spark.rdd.RDD

import scala.annotation.tailrec
import scala.concurrent.Await
import scala.concurrent.duration._


/**
 * __General__ Trainer Implementation.
 *
 * This class trains with help of Training Style and Input Operation.
 *
 * @note This trainer is generalized class. Further implementation, you should see several styles.       
 * @example
 * {{{val net:Network = ...
 *
 *          // Define Manipulation Type. VectorType, AEType, RAEType and URAEType.
 *          val operation = new VectorType(
 *             corrupt = GaussianCorruption(variance = 0.1)
 *          )
 *
 *         // Define Training Style. SingleThreadTrainStyle vs DistBeliefTrainStyle
 *          val style = new SingleThreadTrainStyle(
 *            net = net,
 *            algorithm = new StochasticGradientDescent(l2decay = 0.0001),
 *             make = operation,
 *            param = SimpleTrainingCriteria(miniBatch = 8))
 *
 *         // Define Trainer
 *         val train = new Trainer(
 *            style = style,
 *            stops = StoppingCriteria(maxIter = 100000))
 *
 *         // Do Train
 *         train.train(set, valid)}}}
 *
 * @note To train an autoencoder, you can provide same training set as validation set.
 *
 * @param style __Training style__ that supervises how to train. There are two styles,
 *              one is [[SingleThreadTrainStyle]]
 *              and the other is [[DistBeliefTrainStyle]].
 * @param stops __Stopping Criteria__ that controls the threshold for stopping. (Default : [[StoppingCriteria]])
 * @param name Name used for logging.
 *
 * @tparam IN the type of input.
 *            Currently, [[kr.ac.kaist.ir.deep.fn.ScalarMatrix]] and DAG are supported
 * @tparam OUT the type of output
 *             Currently, [[kr.ac.kaist.ir.deep.fn.ScalarMatrix]] and Null are supported
 */
class Trainer[IN, OUT](val style: TrainStyle[IN, OUT],
                       val stops: StoppingCriteria = StoppingCriteria(),
                       val name: String = "Trainer")
  extends Serializable {
  /** import everything in the style */

  import style._

  /** Logger */
  @transient protected val logger = Logger.getLogger(this.getClass)
  /** Best Parameter History */
  @transient protected var bestParam: IndexedSeq[ScalarMatrix] = null
  /** Best Loss Iteration Number */
  @transient protected var bestIter: Int = 0
  /** Period of validation */
  @transient protected var validationPeriod: Int = 0

  /**
   * Train given sequence, and validate with given sequence.
   *
   * @param set Full Sequence of training set
   * @return Training error (loss)
   */
  def train(set: Seq[Pair]): Scalar = train(set, set)

  /**
   * Train given sequence, and validate with another sequence.
   *
   * @param set Full Sequence of training set
   * @param validation Full Sequence of validation set
   * @return Training error (loss)
   */
  def train(set: Seq[Pair],
            validation: Seq[Pair]): Scalar = {
    setPositiveTrainingReference(set)
    setTestReference(validation)

    validationPeriod = (stops.validationFreq * validationEpoch).toInt

    if (validationPeriod > 0) {
      logger info f"($name) Starts training. "
      logger info f"($name) Every $validationPeriod%5d (${stops.validationFreq * 100}%6.2f%% of TrainingSet), " +
        f"validation process will be submitted."

      saveParams()
      val err = lossOfTraining
      restoreParams()
      printValidation()

      err
    } else {
      logger warn f"($name) Validation Period is zero! Training stopped."
      logger warn f"($name) Please check your validation frequency (currently ${stops.validationFreq}) " +
        f"and training set size (about ${validationEpoch * param.miniBatch})"
      Float.PositiveInfinity
    }
  }

  /**
   * Train using given RDD sequence. 
   *
   * @param set RDD of training set
   */
  def train(set: RDD[Pair]): Scalar = train(set, set)

  /**
   * Train using given RDD sequence. 
   *
   * @param set RDD of training set
   * @param validation RDD of validation set
   */
  def train(set: RDD[Pair], validation: RDD[Pair]): Scalar = {
    setPositiveTrainingReference(set)
    setTestReference(validation)

    validationPeriod = (stops.validationFreq * validationEpoch).toInt

    if (validationPeriod > 0) {
      logger info f"($name) Starts training. "
      logger info f"($name) Every $validationPeriod%5d (${stops.validationFreq * 100}%6.2f%% of TrainingSet), " +
        f"validation process will be submitted."

      saveParams()
      val err = lossOfTraining
      restoreParams()
      printValidation()

      err
    } else {
      logger warn f"($name) Validation Period is zero! Training stopped."
      logger warn f"($name) Please check your validation frequency (currently ${stops.validationFreq}) " +
        f"and training set size (about ${validationEpoch * param.miniBatch})"
      Float.PositiveInfinity
    }
  }

  /**
   * Print validation result into logger
   */
  protected def printValidation() = {
    logger info s"($name) BEST EPOCH ${bestIter + 1}"
    foreachTestSet(5) {
      item ⇒ logger info make.stringOf(net, item)
    }
  }

  /**
   * Store best parameters
   *
   * @param epoch current iteration epoch. (1 iteration = 1 validation freq)
   * @param loss previous loss
   * @param patience current patience, i.e. loop until at least this epoch.
   */
  protected final def saveParams(epoch: Int = 0,
                                 loss: Scalar = Float.MaxValue,
                                 patience: Int = validationPeriod * 5) = {
    bestParam = net.W.copy
    bestIter = epoch
  }

  /**
   * Restore best parameters
   */
  protected final def restoreParams() = {
    // Wait for finish of update, to prohibit race condition.
    if (isUpdateFinished != null) {
      try {
        Await.ready(isUpdateFinished, 5.minutes)
      } catch {
        case _: Throwable ⇒
      }
    }

    net.W := bestParam
  }

  /**
   * Tail Recursive : Train each batch
   *
   * @param epoch current iteration epoch. (1 iteration = 1 validation freq)
   * @param prevloss previous loss
   * @param patience current patience, i.e. loop until at least this epoch.
   * @return Total Loss when train is finished
   */
  @tailrec
  protected final def trainBatch(epoch: Int = 0,
                                 prevloss: Scalar = Float.MaxValue,
                                 patience: Int = 5): Scalar = {
    fetch(epoch)
    batch()
    update(epoch)

    var nPatience = patience
    val iter = epoch / validationPeriod + 1

    val nLoss = if ((epoch + 1) % validationPeriod == 0) {
      // Pending until batch finished
      stopUntilBatchFinished()

      val train = validationError()
      val weight = algorithm loss net.W
      val loss = train + weight
      val improvement = if (prevloss > 0f) loss / prevloss else stops.improveThreshold
      if (improvement < stops.improveThreshold) {
        nPatience = Math.max(patience, iter * (stops.waitAfterUpdate + 1))
        saveParams(epoch, loss, nPatience)

        val impr = 100.0 - improvement * 100.0
        logger info f"($name) # $iter%4d/$nPatience%4d, " +
          f"E + W = $train%.5f + $weight%.5f = $loss%.5f (▼ $impr%8.4f %%) "
        loss
      } else {
        val impr = 100.0 - improvement * 100.0
        if (impr > 0f)
          logger info f"($name) # $iter%4d/$nPatience%4d, " +
            f"DISCARDED: NOT ENOUGH IMPROVEMENT   (▼ $impr%8.4f %%)"
        else
          logger info f"($name) # $iter%4d/$nPatience%4d, " +
            f"DISCARDED: NOT IMPROVED             (△ ${-impr}%8.4f %%)"
        prevloss
      }
    } else {
      prevloss
    }

    if (iter <= stops.maxIter && nPatience >= iter && nLoss >= stops.lossThreshold) {
      trainBatch(epoch + 1, nLoss, nPatience)
    } else {
      if (nLoss < stops.lossThreshold)
        logger info f"($name) # $iter%4d/$nPatience%4d, " +
          f"FINISHED with E + W = $nLoss%.5f [Loss < ${stops.lossThreshold}%.5f]"
      else if (epoch > stops.maxIter)
        logger info f"($name) # $iter%4d/$nPatience%4d, " +
          f"FINISHED with E + W = $nLoss%.5f [Iteration > ${stops.maxIter}%6d]"
      else if (nPatience < epoch)
        logger info f"($name) # $iter%4d/$nPatience%4d, " +
          f"FINISHED with E + W = $nLoss%.5f [NoUpdate ${stops.waitAfterUpdate} Iterations.]"

      nLoss
    }
  }

  /**
   * Do actual training process
   * @return MSE of the training process
   */
  private def lossOfTraining: Float =
    if (param.miniBatch > 0) {
      trainBatch()
    } else {
      fetch(0)
      batch()
      update(0)

      val train = validationError()
      val weight = algorithm loss net.W
      val loss = train + weight
      saveParams(0, loss, 0)

      logger info f"($name) PASSONCE, E + W = $train%.5f + $weight%.5f = $loss%.5f"
      loss
    }

}
