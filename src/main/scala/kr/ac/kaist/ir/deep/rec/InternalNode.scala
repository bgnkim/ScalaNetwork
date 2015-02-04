package kr.ac.kaist.ir.deep.rec

import kr.ac.kaist.ir.deep.fn._
import kr.ac.kaist.ir.deep.train.Corruption

/**
 * __Node__ for internal structure (non-terminal)
 */
class InternalNode(val req: Seq[Node]) extends Node {

  /**
   * Forward computation of DAG
   *
   * @param fn function to be applied
   * @return the result
   */
  override def forward(fn: ScalarMatrix ⇒ ScalarMatrix): ScalarMatrix = {
    val result = req.foldLeft(null.asInstanceOf[ScalarMatrix]) {
      (last, curr) ⇒
        val res = curr.forward(fn)
        if (last != null)
          last row_+ res
        else
          res
    }
    fn(result)
  }

  /**
   * Backward computation of DAG
   *
   * @param err Matrix to be propagated
   * @param fn function to be applied
   * @return Sequence of terminal nodes              
   */
  def backward(err: ScalarMatrix, fn: ScalarMatrix ⇒ ScalarMatrix): Seq[TerminalNode] = {
    val error = fn(err)
    val rSize = error.rows / req.size
    val res = req.foldRight((error, Seq[TerminalNode]())) {
      (curr, pair) ⇒
        val e = pair._1
        val seq = pair._2
        val splited = spliter(e, rSize)
        val seq2 = curr.backward(splited._2, fn)
        //pass left part
        (splited._1, seq2 ++ seq)
    }
    res._2
  }

  /**
   * Corrupt this node
   * *
   * @param corrupt Corruption function to be applied
   * @return Corrupted DAG
   */
  override def through(corrupt: Corruption): Node =
    new InternalNode(req map {
      _ through corrupt
    })
}
