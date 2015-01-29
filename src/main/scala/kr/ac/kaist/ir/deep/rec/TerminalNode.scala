package kr.ac.kaist.ir.deep.rec

import kr.ac.kaist.ir.deep.fn.ScalarMatrix
import kr.ac.kaist.ir.deep.train.Corruption

/**
 * __Node of DAG__ whose position is terminal.
 *
 * This node does not do any computation.
 *
 * @param x original value matrix
 */
class TerminalNode(val x: ScalarMatrix) extends Node {
  var out: ScalarMatrix = x

  /**
   * Forward computation of DAG
   *
   * @param fn function to be applied
   * @return the result
   */
  override def forward(fn: ScalarMatrix ⇒ ScalarMatrix): ScalarMatrix = out

  /**
   * Backward computation of DAG
   *
   * @param err Matrix to be propagated
   * @param fn function to be applied
   * @return Sequence of terminal nodes              
   */
  def backward(err: ScalarMatrix, fn: ScalarMatrix ⇒ ScalarMatrix): Seq[TerminalNode] = {
    out = err
    Seq(this)
  }

  /**
   * Corrupt this node
   * *
   * @param corrupt Corruption function to be applied
   * @return Corrupted DAG
   */
  override def ?(corrupt: Corruption): Node =
    new TerminalNode(corrupt(x))
}
