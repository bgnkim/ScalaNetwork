package kr.ac.kaist.ir.deep.trainer

import kr.ac.kaist.ir.deep.function._
import kr.ac.kaist.ir.deep.network.{AutoEncoder, Network}
import org.apache.log4j.Logger

import scala.annotation.tailrec


/**
 * Trait: Trainer
 */
trait Trainer extends Serializable {
  /** Stopping Criteria */
  protected val stops: StoppingCriteria
  /** Algorithm */
  protected val algorithm: WeightUpdater
  /** Network */
  protected val net: Network
  /** Objective function */
  protected val error: Objective
  /** Corruption function */
  protected val corrupt: Corruption
  /** Training parameters */
  protected val param: TrainingCriteria
  /** Training Set */
  protected var trainingSet: Int ⇒ Seq[(ScalarMatrix, ScalarMatrix)] = null
  /** Validation Set */
  protected var testSet: Int ⇒ Seq[(ScalarMatrix, ScalarMatrix)] = null
  /** Best Parameter History */
  @transient protected var bestParam: Seq[ScalarMatrix] = null
  @transient protected var bestIter: Int = 0
  /** Logger */
  @transient protected val logger = Logger.getLogger(this.getClass)

  /**
   * Implicit weight operation
   * @param w to be applied
   */
  implicit class WeightOp(w: Seq[ScalarMatrix]) extends Serializable {
    /**
     * Sugar: Weight update
     * @param dw is a amount of update
     */
    def -=(dw: Seq[ScalarMatrix]) = algorithm(dw, w)
  }

  /**
   * Train given sequence, and validate with given sequence.
   * @param set to be trained (Random Sequence Generator)
   * @return Training error (loss)
   */
  def train(set: Int ⇒ Seq[(ScalarMatrix, ScalarMatrix)]): Scalar = train(set, set)

  /**
   * Train given sequence, and validate with given sequence.
   * @param set to be trained (Full Seq)
   * @return Training error (loss)
   */
  def train(set: Seq[(ScalarMatrix, ScalarMatrix)]): Scalar = {
    val index = () ⇒ Math.floor(Math.random() * set.size).toInt
    val randomizer = (n: Int) ⇒ (0 until n) map { _ ⇒ set(index())}
    train(randomizer, randomizer)
  }

  /**
   * Train given sequence, and validate with another sequence.
   * @param set to be used for training (Full Seq)
   * @param validation to be used for validation (Full Seq)
   * @return Training error (loss)
   */
  def train(set: Seq[(ScalarMatrix, ScalarMatrix)],
            validation: Seq[(ScalarMatrix, ScalarMatrix)]): Scalar = {
    val index = () ⇒ Math.floor(Math.random() * set.size).toInt
    val randomizer = (n: Int) ⇒ (0 until n) map { _ ⇒ set(index())}
    val topN = (n: Int) ⇒ validation.slice(0, n)
    train(randomizer, topN)
  }

  /**
   * Train given sequence, and validate with another sequence.
   *
   * @param set to be used for training (Randomized Sequence Generator)
   * @param validation to be used for validation (Sequence Generator)
   * @return Training error (loss)
   */
  def train(set: Int ⇒ Seq[(ScalarMatrix, ScalarMatrix)],
            validation: Int ⇒ Seq[(ScalarMatrix, ScalarMatrix)]) = {
    trainingSet = set
    testSet = validation

    saveParams()
    val err = trainBatch()
    restoreParams()
    printValidation()

    err
  }

  /**
   * Train autoencoder with given sequence.
   * @param set to be used for input & reconstruction. (Full Seq)
   * @return Training error (loss)
   */
  def trainAutoencoder(set: Seq[ScalarMatrix]): Scalar = {
    val index = () ⇒ Math.floor(Math.random() * set.size).toInt
    val randomizer = (n: Int) ⇒ (0 until n) map { _ ⇒ set(index())}
    trainAutoencoder(randomizer)
  }

  /**
   * Train autoencoder with given sequence.
   * @param set to be used for input & reconstruction. (Randomized Sequence Generator)
   * @return Training error (loss)
   */
  def trainAutoencoder(set: Int ⇒ Seq[ScalarMatrix]): Scalar = {
    trainingSet = set andThen { seq ⇒ seq map { item ⇒ item → item}}
    testSet = trainingSet

    saveParams()
    val err = trainBatch(isAutoEncoder = true)
    restoreParams()
    printValidation(isAutoEncoder = true)

    err
  }

  /**
   * Calculate validation error
   * @param isAutoEncoder true if it is autoencoder training
   * @return validation error
   */
  protected def validationError(isAutoEncoder: Boolean = false) = {
    val t = testSet(param.validationSize)
    t.foldLeft(0.0) { (err, item) ⇒ {
      val in = item._1
      val out = if (isAutoEncoder) in >>: net else net(in)
      logger.debug(s"IN ${in.mkString} : EXP ${item._2.mkString} = OUT : ${out.mkString}")
      err + error(item._2, out)
    }
    } / t.size
  }

  /**
   * Print validation result
   * @param isAutoEncoder true if it is autoencoder training
   */
  protected def printValidation(isAutoEncoder: Boolean = false) = {
    logger.info(s"BEST ITERATION $bestIter : W = ${net.W map (_.mkString) mkString " | "}")

    val t = testSet(param.validationSize)
    t.par foreach {
      item ⇒ {
        val in = item._1
        val out = if (isAutoEncoder) net.asInstanceOf[AutoEncoder].reconstruct(in) else net(in)
        logger.info(s"IN ${in.mkString} : EXP ${item._2.mkString} = OUT : ${out.mkString}")
      }
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
    net.W := bestParam
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
    fetch(iter)
    batch()
    update(iter)

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

  /**
   * Fetch weights 
   * @param iter is current iteration
   */
  protected def fetch(iter: Int): Unit

  /**
   * Do mini-batch
   */
  protected def batch(): Unit

  /**
   * Send update of weights
   * @param iter is current iteration
   */
  protected def update(iter: Int): Unit
}
