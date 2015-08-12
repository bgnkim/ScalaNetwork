package kr.ac.kaist.ir.deep.train

import java.text.SimpleDateFormat
import java.util.Date

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
 *          // Define Manipulation Type. VectorType, AEType, RAEType, StandardRAEType, URAEType, and StringToVectorType.
 *          val style = new SingleThreadTrainStyle(
 *            net = net,
 *            algorithm = new StochasticGradientDescent(l2decay = 0.0001),
 *             make = operation,
 *             param = SimpleTrainingCriteria(miniBatchFraction = 0.01))
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

  @transient private final val dateFormatter = new SimpleDateFormat("MM/dd HH:mm:ss")
  /** Logger */
  @transient protected val logger = Logger.getLogger(this.getClass)
  /** Best Parameter History */
  @transient protected var bestParam: IndexedSeq[ScalarMatrix] = null
  /** Best Loss Iteration Number */
  @transient protected var bestIter: Int = 0
  /** Period of validation */
  @transient protected var validationPeriod: Int = 0
  /** Get command line column width */
  @transient protected var columns = try {
    System.getenv("COLUMNS").toInt
  } catch {
    case _: Throwable ⇒ 80
  }
  /** Finish time of last iteration */
  @transient protected var startAt: Long = _

  /**
   * Train given sequence, and validate with given sequence.
   *
   * @param set Full Sequence of training set
   * @return Training error (loss)
   */
  def train(set: Seq[Pair]): (Scalar, Scalar, Scalar) = train(set, set)

  /**
   * Train given sequence, and validate with another sequence.
   *
   * @param set Full Sequence of training set
   * @param validation Full Sequence of validation set
   * @return Training error (loss)
   */
  def train(set: Seq[Pair],
            validation: Seq[Pair]): (Scalar, Scalar, Scalar) = {
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
      logger warn f"($name) Maybe because miniBatchFraction value is too large. Please check."
      (Float.PositiveInfinity, Float.PositiveInfinity, Float.PositiveInfinity)
    }
  }

  /**
   * Train using given RDD sequence. 
   *
   * @param set RDD of training set
   */
  def train(set: RDD[Pair]): (Scalar, Scalar, Scalar) = train(set, set)

  /**
   * Train using given RDD sequence. 
   *
   * @param set RDD of training set
   * @param validation RDD of validation set
   */
  def train(set: RDD[Pair], validation: RDD[Pair]): (Scalar, Scalar, Scalar) = {
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
      logger warn f"($name) Maybe because miniBatchFraction value is too large. Please check."
      (Float.PositiveInfinity, Float.PositiveInfinity, Float.PositiveInfinity)
    }
  }

  /**
   * Print validation result into logger
   */
  protected def printValidation() = {
    logger info s"($name) BEST ITERATION : $bestIter"
    foreachTestSet(5) {
      item ⇒ logger info make.stringOf(net, item)
    }
  }

  /**
   * Store best parameters
   *
   * @param iteration current iteration. (1 iteration = 1 validation freq)
   * @param loss previous loss
   * @param patience current patience, i.e. loop until at least this epoch.
   */
  protected final def saveParams(iteration: Int = 0,
                                 loss: Scalar = Float.MaxValue,
                                 patience: Int = validationPeriod * 5) = {
    bestParam = net.W.copy
    bestIter = iteration
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
   * @param prevEloss previous loss (Evaluation)
   * @param prevWloss previous loss (Weight)
   * @param patience current patience, i.e. loop until at least this epoch.
   * @return (Evaluation, Weight, Total) Loss when train is finished
   */
  @tailrec
  protected final def trainBatch(epoch: Int = 0,
                                 prevEloss: Scalar = Float.MaxValue,
                                 prevWloss: Scalar = Float.MaxValue,
                                 patience: Int = 5): (Scalar, Scalar, Scalar) = {
    fetch(epoch)
    batch()
    update(epoch)

    var nPatience = patience
    val iter = epoch / validationPeriod + 1

    val prevloss = prevEloss + prevWloss
    val nLoss = if ((epoch + 1) % validationPeriod == 0) {
      // Pending until batch finished
      stopUntilBatchFinished()

      val train = validationError()
      val weight = algorithm loss net.W
      val loss = train + weight
      val improvement = if (prevloss > 0f) loss / prevloss else stops.improveThreshold
      if (improvement < stops.improveThreshold) {
        nPatience = Math.min(Math.max(patience, iter * (stops.waitAfterUpdate + 1)), stops.maxIter)
        saveParams(iter, loss, nPatience)

        printProgress(iter, nPatience, train, weight, improved = true)
        (train, weight, loss)
      } else {
        printProgress(iter, nPatience, prevEloss, prevWloss, improved = false)
        (prevEloss, prevWloss, prevloss)
      }
    } else {
      (prevEloss, prevWloss, prevloss)
    }

    if (iter <= nPatience && (nLoss._3 >= stops.lossThreshold || iter < 5)) {
      trainBatch(epoch + 1, nLoss._1, nLoss._2, nPatience)
    } else {
      if (nLoss._3 < stops.lossThreshold)
        logger info f"($name) # $iter%4d/$nPatience%4d, " +
          f"FINISHED with E + W = ${nLoss._3}%.5f [Loss < ${stops.lossThreshold}%.5f]"
      else if (iter > stops.maxIter)
        logger info f"($name) # $iter%4d/$nPatience%4d, " +
          f"FINISHED with E + W = ${nLoss._3}%.5f [Iteration > ${stops.maxIter}%6d]"
      else if (nPatience < iter)
        logger info f"($name) # $iter%4d/$nPatience%4d, " +
          f"FINISHED with E + W = ${nLoss._3}%.5f [NoUpdate after $bestIter%6d]"

      nLoss
    }
  }

  private def printProgress(iter: Int, patience: Int, eLoss: Float, wLoss: Float, improved: Boolean) = {
    val wait = patience / stops.maxIter.toFloat
    val header = f"\033[4m$name\033[24m $iter%4d/$patience%4d \033[0m["
    val impr = if (improved) "IMPROVED" else f" @ $bestIter%4d "
    val footer = f" E + W = $eLoss%7.5f + $wLoss%7.5f $impr"

    val buf = new StringBuilder(s"\033[2A\033[${columns}D\033[2K \033[1;33m$header\033[46;36m")
    val total = columns - header.length - footer.length + 10
    val len = Math.floor(wait * total).toInt
    val step = Math.floor(iter / stops.maxIter.toFloat * total).toInt
    buf.append(" " * step)
    buf.append("\033[49m")
    buf.append(" " * (len - step))
    buf.append("\033[0m]\033[34m")
    if (total > len) buf.append(s"\033[${total - len}C")
    buf.append(s"$footer\033[0m")

    val now = System.currentTimeMillis()
    val remainA = (now - startAt) / iter * patience
    val etaA = startAt + remainA
    val calA = dateFormatter.format(new Date(etaA))
    val remainB = (now - startAt) / iter * stops.maxIter
    val etaB = startAt + remainB
    val calB = dateFormatter.format(new Date(etaB))

    buf.append(f"\n\033[2K Estimated Finish Time : $calA \t ~ $calB")

    println(buf.result())
  }

  /**
   * Do actual training process
   * @return MSE of the training process
   */
  private def lossOfTraining: (Scalar, Scalar, Scalar) =
    if (param.miniBatchFraction > 0) {
      println("Start training...\n Estimated Time: NONE")
      startAt = System.currentTimeMillis()
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
      (train, weight, loss)
    }

}
