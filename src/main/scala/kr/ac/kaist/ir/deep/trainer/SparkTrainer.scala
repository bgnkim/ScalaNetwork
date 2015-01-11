package kr.ac.kaist.ir.deep.trainer

import kr.ac.kaist.ir.deep.function._
import kr.ac.kaist.ir.deep.network.{AutoEncoder, Network}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import scala.annotation.tailrec

/**
 * Trainer : Semi-DistBelief Style, Spark-based.
 *
 * Unlike with DistBelief, this trainer do updates and fetch by "master" not the "workers".
 * Real DistBelief implementation is planned.
 *
 * @param net to be trained
 * @param algorithm to be applied
 * @param sc is a spark context that network will be distributed
 * @param error to compute loss
 * @param corrupt to corrupt input
 * @param param of training criteria
 * @param stops of stopping criteria
 */
class SparkTrainer(private val net: Network,
                   private val algorithm: WeightUpdater,
                   @transient private val sc: SparkContext,
                   private val error: Objective = SquaredErr,
                   private val corrupt: Corruption = NoCorruption,
                   private val param: DistBeliefCriteria = DistBeliefCriteria(),
                   protected override val stops: StoppingCriteria = StoppingCriteria(),
                   private val debug: Boolean = false)
  extends Trainer {
  /** Spark distributed networks */
  @transient protected val networks: RDD[Network] = sc.makeRDD(net copy param.numCores)
  /** Training Set */
  protected var trainingSet: Int ⇒ Seq[(ScalarMatrix, ScalarMatrix)] = null
  /** Validation Set */
  @transient protected var testSet: Int ⇒ Seq[(ScalarMatrix, ScalarMatrix)] = null
  /** Best Parameter History */
  @transient protected var bestParam: Seq[ScalarMatrix] = null
  @transient protected var bestIter: Int = 0

  /**
   * Train given sequence, and validate with another sequence.
   *
   * @param set to be used for training (Randomized Sequence Generator)
   * @param validation to be used for validation (Sequence Generator)
   * @return Training error (loss)
   */
  override def train(set: Int ⇒ Seq[(ScalarMatrix, ScalarMatrix)], validation: Int ⇒ Seq[(ScalarMatrix, ScalarMatrix)]) = {
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
  def train(set: RDD[(ScalarMatrix, ScalarMatrix)]): Scalar = train(set.takeSample(true, _))

  /**
   * Train autoencoder with given sequence.
   * @param set to be used for input & reconstruction. (Randomized Sequence Generator)
   * @return Training error (loss)
   */
  override def trainAutoencoder(set: Int ⇒ Seq[ScalarMatrix]): Scalar = {
    trainingSet = set andThen { seq ⇒ seq map { item ⇒ item → item}}
    testSet = trainingSet

    saveParams()
    val err = trainBatch(isAutoEncoder = true)
    restoreParams()
    printValidation(isAutoEncoder = true)

    err
  }

  /**
   * Store best parameters
   */
  protected final def saveParams() = {
    bestParam = net.W map {
      _.copy
    }
  }

  /**
   * Restore best parameters
   */
  protected final def restoreParams() = {
    bestParam.indices.par foreach {
      id ⇒ net.W(id) := bestParam(id)
    }
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
      if (debug) {
        println(s"IN ${in.mkString} : EXP ${item._2.mkString} = OUT : ${out.mkString}")
      }
      err + error(item._2, out)
    }
    } / t.size
  }

  /**
   * Print validation result
   * @param isAutoEncoder true if it is autoencoder training
   */
  protected def printValidation(isAutoEncoder: Boolean = false) = {
    println(s"BEST ITERATION $bestIter : W = ${net.W map (_.mkString) mkString " | "}")

    val t = testSet(param.validationSize)
    t.par foreach {
      item ⇒ {
        val in = item._1
        val out = if (isAutoEncoder) net.asInstanceOf[AutoEncoder].reconstruct(in) else net(in)
        println(s"IN ${in.mkString} : EXP ${item._2.mkString} = OUT : ${out.mkString}")
      }
    }
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
    if (iter % param.fetchStep == 0) {
      val weights = sc.broadcast(net.W)
      networks foreach (net ⇒ {
        val w = net.W
        w.indices foreach {
          id ⇒ w(id) := weights.value(id)
        }
      })
      weights.destroy()
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
    if (iter % param.updateStep == 0) {
      val dWUpdate = networks.aggregate(Seq[ScalarMatrix]())({
        (seq, copiedNet) ⇒
          if (seq.isEmpty) copiedNet.dW
          else
            seq.indices map {
              id ⇒ copiedNet.dW(id) + seq(id)
            }
      }, {
        case (dW1, dW2) if dW2.isEmpty ⇒ dW1
        case (dW1, dW2) if dW1.isEmpty ⇒ dW2
        case (dW1, dW2) ⇒
          dW1.indices map {
            id ⇒ dW1(id) + dW2(id)
          }
        case _ ⇒ Seq[ScalarMatrix]()
      })

      algorithm(dWUpdate, net.W)
    }

    var nPatience = patience

    val nLoss = if ((iter + 1) % stops.validationFreq == 0) {
      if (debug) {
        println(s"ITERATION $iter : W = ${net.W map (_.mkString) mkString " | "}")
      }
      val train = validationError(isAutoEncoder)
      val weight = algorithm loss net.W
      if (train + weight < prevloss * stops.improveThreshold) {
        nPatience = Math.max(patience, iter * stops.patienceStep)
        bestIter = iter
        saveParams()
        println(f"Iteration $iter%6d, Validation = $train%.5f, WeightLoss = $weight%.5f")
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
      println(f"Finished $iter%6d, Error = $nLoss%.5f")
      nLoss
    }
  }
}
