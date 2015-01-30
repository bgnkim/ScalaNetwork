package kr.ac.kaist.ir.deep.rec

import kr.ac.kaist.ir.deep.fn._
import kr.ac.kaist.ir.deep.train.Corruption

/**
 * __Graph__: Directed Acyclic Graph
 *
 * This class collects all the output results, and just concatenate it.
 * Therefore this class does not directly apply given functions.
 */
class DAG(val finals: Seq[Node]) extends Node {
  /**
   * Forward computation of DAG
   *
   * @param fn function to be applied
   * @return the result
   */
  override def forward(fn: ScalarMatrix ⇒ ScalarMatrix): ScalarMatrix = {
    finals.foldLeft(null.asInstanceOf[ScalarMatrix]) {
      (last, curr) ⇒
        val res = curr.forward(fn)
        if (last != null)
          last row_+ res
        else
          res
    }
  }

  /**
   * Backward computation of DAG
   *
   * @param err Matrix to be propagated
   * @param fn function to be applied
   * @return Sequence of terminal nodes              
   */
  def backward(err: ScalarMatrix,
               fn: ScalarMatrix ⇒ ScalarMatrix): Seq[TerminalNode] = {
    val rSize = err.rows / finals.size
    val res = finals.foldRight((err, Seq[TerminalNode]())) {
      (curr, pair) ⇒
        val e = pair._1
        val seq = pair._2
        val splited = spliter(e, rSize)
        val seq2 = curr.backward(splited._1, fn)
        (splited._2, seq2 ++ seq)
    }
    res._2
  }

  /**
   * Corrupt this node
   * *
   * @param corrupt Corruption function to be applied
   * @return Corrupted DAG
   */
  override def through(corrupt: Corruption): DAG =
    new DAG(finals map {
      _ through corrupt
    })
}