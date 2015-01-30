package kr.ac.kaist.ir.deep.rec

import kr.ac.kaist.ir.deep.fn.ScalarMatrix
import kr.ac.kaist.ir.deep.train.Corruption

/**
 * __Trait__ that describes a node in DAG.
 */
trait Node extends Serializable {
  /**
   * Matrix spliter
   */
  protected val spliter = (obj: ScalarMatrix, rightSize: Int) ⇒ {
    val startFrom = obj.rows - rightSize
    (obj(0 until startFrom, ::), obj(startFrom to -1, ::))
  }

  /**
   * Forward computation of DAG
   *
   * @param fn function to be applied
   * @return the result
   */
  def forward(fn: ScalarMatrix ⇒ ScalarMatrix): ScalarMatrix

  /**
   * Backward computation of DAG
   *
   * @param err Matrix to be propagated
   * @param fn function to be applied
   * @return Sequence of terminal nodes              
   */
  def backward(err: ScalarMatrix, fn: ScalarMatrix ⇒ ScalarMatrix): Seq[TerminalNode]

  /**
   * Corrupt this node
   * *
   * @param corrupt Corruption function to be applied
   * @return Corrupted DAG
   */
  def through(corrupt: Corruption): Node
}
