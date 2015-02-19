package kr.ac.kaist.ir.deep.rec

import kr.ac.kaist.ir.deep.fn._
import kr.ac.kaist.ir.deep.train.Corruption

/**
 * __Node__ for internal structure (non-terminal)
 */
class BinaryTree(val left: Node, right: Node) extends Node {

  /**
   * Forward computation of Binary Tree
   *
   * @param fn function to be applied
   * @return the result
   */
  override def forward(fn: ScalarMatrix ⇒ ScalarMatrix): ScalarMatrix = {
    val leftMatx = left.forward(fn)
    val rightMatx = right.forward(fn)
    fn(leftMatx row_+ rightMatx)
  }

  /**
   * Backward computation of Binary Tree
   *
   * @param err Matrix to be propagated
   * @param fn function to be applied
   * @return Sequence of terminal nodes              
   */
  def backward(err: ScalarMatrix, fn: ScalarMatrix ⇒ ScalarMatrix): Seq[Leaf] = {
    val error = fn(err)
    val rSize = error.rows / 2

    val seqLeft = left.backward(error(0 until rSize, ::), fn)
    val seqRight = right.backward(error(rSize to -1, ::), fn)
    seqLeft ++ seqRight
  }

  /**
   * Corrupt this node
   * *
   * @param corrupt Corruption function to be applied
   * @return Corrupted Binary Tree
   */
  override def through(corrupt: Corruption): Node =
    new BinaryTree(left through corrupt, right through corrupt)
}
