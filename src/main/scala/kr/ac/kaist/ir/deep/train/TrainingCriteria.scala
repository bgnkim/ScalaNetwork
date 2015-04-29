package kr.ac.kaist.ir.deep.train

/**
 * __Trait__ that describes Training Criteria
 */
trait TrainingCriteria extends Serializable {
  /** Size of mini-batch.
    * If below or equal to zero, then this indicates no batch training (i.e. just go through once.) */
  val miniBatch: Int
}
