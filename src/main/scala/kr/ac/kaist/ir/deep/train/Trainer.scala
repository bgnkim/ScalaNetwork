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
 *
 * @tparam IN the type of input.
 *            Currently, [[ScalarMatrix]] and DAG are supported
 * @tparam OUT the type of output
 *             Currently, [[ScalarMatrix]] and Null are supported
 */
class Trainer[IN, OUT](protected val style: TrainStyle[IN, OUT],
                       protected val stops: StoppingCriteria = StoppingCriteria())
  extends Serializable {
  /** import everything in the style */

  import style._

  /** Logger */
  @transient protected val logger = Logger.getLogger(this.getClass)
  /** Validation Set */
  protected var testSet: Int ⇒ Array[Pair] = null
  /** Best Parameter History */
  @transient protected var bestParam: IndexedSeq[ScalarMatrix] = null
  /** Best Loss Iteration Number */
  @transient protected var bestIter: Int = 0

  /**
   * Train given sequence, and validate with given sequence.
   *
   * @param set __Random Sequence Generator__ of training set
   * @return Training error (loss)
   */
  def train(set: Int ⇒ Array[Pair]): Scalar = train(set, set)

  /**
   * Train given sequence, and validate with given sequence.
   *
   * @param set Full Sequence of training set
   * @return Training error (loss)
   */
  def train(set: Seq[Pair]): Scalar = {
    val index = () ⇒ Math.floor(Math.random() * set.size).toInt
    val randomizer = (n: Int) ⇒ {
      val array = new Array[Pair](n)
      var i = 0
      while (i < n) {
        array.update(i, set(index()))
        i += 1
      }
      array
    }
    train(randomizer, randomizer)
  }

  /**
   * Train given sequence, and validate with another sequence.
   *
   * @param set Full Sequence of training set
   * @param validation Full Sequence of validation set
   * @return Training error (loss)
   */
  def train(set: Seq[Pair],
            validation: Seq[Pair]): Scalar = {
    val index = () ⇒ Math.floor(Math.random() * set.size).toInt
    val randomizer = (n: Int) ⇒ {
      val array = new Array[Pair](n)
      var i = 0
      while (i < n) {
        array.update(i, set(index()))
        i += 1
      }
      array
    }
    val topN = (n: Int) ⇒ validation.slice(0, n).toArray
    train(randomizer, topN)
  }

  /**
   * Train given sequence, and validate with another sequence.
   *
   * @param set __Randomized Sequence Generator__ of training set
   * @param validation Sequence Generator of validation set
   * @return Training error (loss)
   */
  def train(set: Int ⇒ Array[Pair],
            validation: Int ⇒ Array[Pair]) = {
    trainingSet = set
    testSet = validation

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
  def train(set: RDD[Pair]): Scalar = {
    train({
      val cached = set.cache()
      x: Int ⇒ cached.takeSample(withReplacement = true, num = x)
    })
  }

  /**
   * Train using given RDD sequence. 
   *
   * @param set RDD of training set
   * @param validation RDD of validation set
   */
  def train(set: RDD[Pair], validation: RDD[Pair]): Scalar = {
    train({
      x: Int ⇒ set.takeSample(withReplacement = true, num = x)
    }, {
      x: Int ⇒ validation.takeSample(withReplacement = true, num = x)
    })
  }

  /**
   * Calculate validation error
   *
   * @return validation error
   */
  protected def validationError() = {
    val t = testSet(param.validationSize)
    make.lossOf(net, t) / t.size
  }

  /**
   * Print validation result into logger
   */
  protected def printValidation() = {
    logger info s"BEST ITERATION $bestIter : W = ${net.W map (_.mkString) mkString " | "}"
    testSet(5).map {
      item ⇒
        logger info make.stringOf(net, item)
    }
  }

  /**
   * Store best parameters
   */
  protected final def saveParams() = {
    bestParam = net.W.copy
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
   * @param iter current iteration
   * @param prevloss previous loss
   * @param patience current patience
   * @return Total Loss when train is finished
   */
  @tailrec
  protected final def trainBatch(iter: Int = 0,
                                 prevloss: Double = Double.MaxValue,
                                 patience: Int = stops.patience): Scalar = {
    fetch(iter)
    batch()
    update(iter)

    var nPatience = patience

    val nLoss = if ((iter + 1) % stops.validationFreq == 0) {
      logger debug s"ITERATION $iter : W = ${net.W map (_.mkString) mkString " | "}"
      val train = validationError()
      val weight = algorithm loss net.W
      if (train + weight < prevloss * stops.improveThreshold) {
        nPatience = Math.max(patience, iter * stops.patienceStep)
        bestIter = iter
        saveParams()
        logger info f"Iteration $iter%6d, Validation = $train%.5f, WeightLoss = $weight%.5f"
        train + weight
      } else {
        prevloss
      }
    } else {
      prevloss
    }

    if (iter < stops.maxIter && nPatience > iter && nLoss > stops.lossThreshold) {
      trainBatch(iter + 1, nLoss, nPatience)
    } else {
      logger.info(f"Finished $iter%6d, Error = $nLoss%.5f")
      nLoss
    }
  }

}