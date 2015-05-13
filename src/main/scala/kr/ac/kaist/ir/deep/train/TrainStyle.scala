package kr.ac.kaist.ir.deep.train

import kr.ac.kaist.ir.deep.fn._
import kr.ac.kaist.ir.deep.network.Network
import org.apache.log4j.Logger
import org.apache.spark.rdd.RDD

import scala.concurrent.Future

/**
 * __Trait__ that describes style of training
 *
 * This trait controls how to train, i.e. __Single-threaded__ or __Distributed__.
 *
 * @tparam IN the type of input
 * @tparam OUT the type of output
 */
trait TrainStyle[IN, OUT] extends Serializable {
  /** Training Pair Type */
  type Pair = (IN, OUT)
  /** Sampler Type */
  type Sampler = Int ⇒ Seq[OUT]
  /** Training parameters */
  val param: TrainingCriteria
  /** Network */
  val net: Network
  /** Algorithm */
  val algorithm: WeightUpdater
  /** Set of input manipulations */
  val make: ManipulationType[IN, OUT]
  /** Logger */
  @transient protected val logger = Logger.getLogger(this.getClass)
  /** number of epochs for iterating one training set */
  var validationEpoch: Int = 0

  /**
   * Calculate validation error
   *
   * @return validation error
   */
  def validationError(): Scalar

  /**
   * Iterate over given number of test instances
   * @param n number of random sampled instances
   * @param fn iteratee function
   */
  def foreachTestSet(n: Int)(fn: Pair ⇒ Unit): Unit

  /**
   * Set training instances 
   * @param set Sequence of training set
   */
  def setPositiveTrainingReference(set: Seq[Pair]): Unit

  /**
   * Set training instances
   * @param set RDD of training set
   */
  def setPositiveTrainingReference(set: RDD[Pair]): Unit

  /**
   * Set testing instances 
   * @param set Sequence of testing set
   */
  def setTestReference(set: Seq[Pair]): Unit

  /**
   * Set testing instances
   * @param set RDD of testing set
   */
  def setTestReference(set: RDD[Pair]): Unit
  
  /**
   * Fetch weights
   *
   * @param iter current iteration
   */
  def fetch(iter: Int): Unit

  /**
   * Do mini-batch
   */
  def batch(): Unit

  /**
   * Send update of weights
   *
   * @param iter current iteration
   */
  def update(iter: Int): Unit

  /**
   * Indicates whether the asynchronous update is finished or not.
   *
   * @return future object of update
   */
  def isUpdateFinished: Future[_] = null

  /**
   * Non-blocking pending, until all assigned batches are finished
   */
  def stopUntilBatchFinished(): Unit = {}

  /**
   * Implicit weight operation
   *
   * @param w Sequence of weight to be applied
   */
  implicit class WeightOp(w: IndexedSeq[ScalarMatrix]) extends Serializable {
    /**
     * Sugar: Weight update
     *
     * @param dw A amount of update i.e. __ΔWeight__
     */
    def -=(dw: IndexedSeq[ScalarMatrix]) = algorithm(dw, w)
  }

}
