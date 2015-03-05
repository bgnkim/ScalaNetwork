package kr.ac.kaist.ir.deep.rec

import kr.ac.kaist.ir.deep.fn.ScalarMatrix
import kr.ac.kaist.ir.deep.train.Corruption

/**
 * __Node of BinaryTree__ whose position is terminal.
 *
 * This node does not do any computation.
 *
 * @param x original value matrix
 */
class Leaf(val x: ScalarMatrix) extends Node {
  var out: ScalarMatrix = x

  /**
   * Forward computation of Binary Tree
   *
   * @param fn function to be applied
   * @return the result
   */
  override def forward(fn: ScalarMatrix ⇒ ScalarMatrix): ScalarMatrix = out

  /**
   * Backward computation of Binary Tree
   *
   * @param err Matrix to be propagated
   * @param fn function to be applied
   * @return Sequence of terminal nodes              
   */
  def backward(err: ScalarMatrix, fn: ScalarMatrix ⇒ ScalarMatrix): Seq[Leaf] = {
    out = err
    Seq(this)
  }

  /**
   * Corrupt this node
   * *
   * @param corrupt Corruption function to be applied
   * @return Corrupted Binary Tree
   */
  override def through(corrupt: Corruption): Node =
    new Leaf(corrupt(x))

  /**
   * Replace wildcard node
   * @param resolve Wildcard Resolver function
   * @return new Node without wildcard
   */
  override def ?(resolve: (Int) ⇒ Node): Node = this
}
