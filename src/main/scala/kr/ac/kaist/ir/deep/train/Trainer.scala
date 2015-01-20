package kr.ac.kaist.ir.deep.train

import kr.ac.kaist.ir.deep.fn._
import kr.ac.kaist.ir.deep.train.op.{InputOp, ScalarVector}
import kr.ac.kaist.ir.deep.train.style.TrainStyle
import org.apache.log4j.Logger
import org.apache.spark.rdd.RDD

import scala.annotation.tailrec


/**
 * General Trainer
 *
 * @param style supervises how to train. There are two styles, 
 *              one is [[kr.ac.kaist.ir.deep.train.style.SingleThreadTrainStyle]]
 *              and the other is [[kr.ac.kaist.ir.deep.train.style.DistBeliefTrainStyle]].
 * @param make supervises how to manipulate input as matrices.
 *             This also controls how to compute actual network.             
 * @param stops controls the threshold for stopping. (Default : [[StoppingCriteria]])
 *
 * @tparam IN is the type of input. 
 *            Currently, [[kr.ac.kaist.ir.deep.fn.ScalarMatrix]] and [[kr.ac.kaist.ir.deep.rec.VectorTree]] are supported
 */
class Trainer[IN](protected val style: TrainStyle[IN],
                  protected[train] val make: InputOp[IN] = new ScalarVector(),
                  protected val stops: StoppingCriteria = StoppingCriteria())
  extends Serializable {
  /** import everything in the style */

  import style._

  /** Logger */
  @transient protected val logger = Logger.getLogger(this.getClass)
  /** Validation Set */
  protected var testSet: Int ⇒ Seq[(IN, ScalarMatrix)] = null
  /** Best Parameter History */
  @transient protected var bestParam: Seq[ScalarMatrix] = null
  @transient protected var bestIter: Int = 0

  /**
   * Train given sequence, and validate with given sequence.
   * @param set to be trained (Random Sequence Generator)
   * @return Training error (loss)
   */
  def train(set: Int ⇒ Seq[(IN, ScalarMatrix)]): Scalar = train(set, set)

  /**
   * Train given sequence, and validate with given sequence.
   * @param set to be trained (Full Seq)
   * @return Training error (loss)
   */
  def train(set: Seq[(IN, ScalarMatrix)]): Scalar = {
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
  def train(set: Seq[(IN, ScalarMatrix)],
            validation: Seq[(IN, ScalarMatrix)]): Scalar = {
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
  def train(set: Int ⇒ Seq[(IN, ScalarMatrix)],
            validation: Int ⇒ Seq[(IN, ScalarMatrix)]) = {
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
   * @param set to be used for training
   */
  def train(set: RDD[(IN, ScalarMatrix)]): Scalar = {
    set.cache()
    train(set.takeSample(true, _))
  }

  /**
   * Train using given RDD sequence. 
   * @param set to be used for training
   * @param validation to be used for validation
   */
  def train(set: RDD[(IN, ScalarMatrix)], validation: RDD[(IN, ScalarMatrix)]): Scalar = {
    set.cache()
    validation.cache()
    train(set.takeSample(true, _), validation.takeSample(true, _))
  }
  
  /**
   * Calculate validation error
   * @param isAutoEncoder true if it is autoencoder training
   * @return validation error
   */
  protected def validationError(isAutoEncoder: Boolean = false) = {
    val t = testSet(param.validationSize)
    t.foldLeft(0.0) {
      (err, item) ⇒
        val out = make onewayTrip(net, item._1)
        logger.debug(s"${make stringOf item} = OUT : ${out.mkString}")
        err + (make error(item._2, out))
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
      item ⇒
        val out = make onewayTrip(net, item._1)
        logger.info(s"${make stringOf item} = OUT : ${out.mkString}")
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
   * @return Total Loss when train is finished
   */
  @tailrec
  protected final def trainBatch(iter: Int = 0,
                                 prevloss: Double = Double.MaxValue,
                                 patience: Int = stops.patience): Scalar = {
    fetch(iter)
    batch(make)
    update(iter)

    var nPatience = patience

    val nLoss = if ((iter + 1) % stops.validationFreq == 0) {
      logger.debug(s"ITERATION $iter : W = ${net.W map (_.mkString) mkString " | "}")
      val train = validationError()
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
      trainBatch(iter + 1, nLoss, nPatience)
    } else {
      logger.info(f"Finished $iter%6d, Error = $nLoss%.5f")
      nLoss
    }
  }

}
