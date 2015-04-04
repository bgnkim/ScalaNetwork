package kr.ac.kaist.ir.deep.train

/**
 * __Trait__ that describes Training Criteria
 */
trait TrainingCriteria extends Serializable {
  /** Size of mini-batch */
  val miniBatch: Int
  /** Sampling rate of negative examples per positive examples. */
  val negSamplingRatio: Int
}
