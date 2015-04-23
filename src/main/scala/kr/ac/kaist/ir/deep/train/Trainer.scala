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
    logger info f"($name) Starts training. "
    logger info f"Every $validationPeriod%5d (${stops.validationFreq * 100}%6.2f%% of TrainingSet), validation process will be submitted."

    saveParams()
    val err = trainBatch()
    restoreParams()
    printValidation()

    err
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
    logger info f"($name) Starts training. "
    logger info f"Every $validationPeriod%5d (${stops.validationFreq * 100}%6.2f%% of TrainingSet), validation process will be submitted."

    saveParams()
    val err = trainBatch()
    restoreParams()
    printValidation()

    err
  }


  /**
   * Set negative sampling method.
   *
   * @param set all training outputs that will be used for negative training
   */
  def setNegativeTrainingReference(set: RDD[OUT]) = {
    require(make.isInstanceOf[VectorType], "Currently, only 'VectorType' manipulation supports negative sampling.")
    style.setNegativeTrainingReference(set)
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
                                 patience: Int = validationPeriod * 5): Scalar = {
    fetch(epoch)
    batch()
    update(epoch)

    var nPatience = patience

    val nLoss = if ((epoch + 1) % validationPeriod == 0) {
      // Pending until batch finished
      stopUntilBatchFinished()

      logger debug s"($name) Ep ${epoch + 1} : W = ${net.W map (_.mkString) mkString " | "}"
      val train = validationError()
      val weight = algorithm loss net.W
      val loss = train + weight
      val improvement = if (prevloss > 0f) loss / prevloss else 0.0f
      if (improvement < stops.improveThreshold) {
        nPatience = Math.max(patience, (epoch + 1) * (stops.waitAfterUpdate + 1))
        saveParams(epoch, loss, nPatience)

        val impr = 100.0 - improvement * 100.0
        logger info f"($name) # ${epoch + 1}%6d/$nPatience%6d, E + W = $train%.5f + $weight%.5f = $loss%.5f (▼ $impr%8.4f %%) "
        loss
      } else {
        val impr = 100.0 - improvement * 100.0
        if (impr > 0f)
          logger info f"($name) # ${epoch + 1}%6d/$nPatience%6d, DISCARDED: NOT ENOUGH IMPROVEMENT   (▼ $impr%8.4f %%)"
        else
          logger info f"($name) # ${epoch + 1}%6d/$nPatience%6d, DISCARDED: NOT IMPROVED             (△ ${-impr}%8.4f %%)"
        prevloss
      }
    } else {
      prevloss
    }

    if (epoch <= stops.maxIter && nPatience >= epoch && nLoss >= stops.lossThreshold) {
      trainBatch(epoch + 1, nLoss, nPatience)
    } else {
      if (nLoss < stops.lossThreshold)
        logger info f"($name) # ${epoch + 1}%6d/$nPatience%6d, FINISHED with E + W = $nLoss%.5f [Loss < ${stops.lossThreshold}%.5f]"
      else if (epoch > stops.maxIter)
        logger info f"($name) # ${epoch + 1}%6d/$nPatience%6d, FINISHED with E + W = $nLoss%.5f [Epoch > ${stops.maxIter}%6d]"
      else if (nPatience < epoch)
        logger info f"($name) # ${epoch + 1}%6d/$nPatience%6d, FINISHED with E + W = $nLoss%.5f [NoUpdate ${stops.waitAfterUpdate} × $bestIter Ep.]"

      nLoss
    }
  }

}
