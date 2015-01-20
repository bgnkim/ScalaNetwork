package kr.ac.kaist.ir.deep.train

/**
 * __Trait__ that describes Training Criteria
 */
trait TrainingCriteria extends Serializable {
  /** Size of mini-batch */
  val miniBatch: Int
  /** Size of validation */
  val validationSize: Int
}
