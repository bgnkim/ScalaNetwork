package kr.ac.kaist.ir.deep.rec

import kr.ac.kaist.ir.deep.fn.ScalarMatrix
import kr.ac.kaist.ir.deep.train.Corruption

/**
 * __Trait__ that describes a node in BinaryTree.
 */
trait Node extends Serializable {
  /**
   * Forward computation of Binary Tree
   *
   * @param fn function to be applied
   * @return the result
   */
  def forward(fn: ScalarMatrix ⇒ ScalarMatrix): ScalarMatrix

  /**
   * Backward computation of Binary Tree
   *
   * @param err Matrix to be propagated
   * @param fn function to be applied
   * @return Sequence of terminal nodes              
   */
  def backward(err: ScalarMatrix, fn: ScalarMatrix ⇒ ScalarMatrix): Seq[Leaf]

  /**
   * Corrupt this node
   * *
   * @param corrupt Corruption function to be applied
   * @return Corrupted Binary Tree
   */
  def through(corrupt: Corruption): Node
}
