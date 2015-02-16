package kr.ac.kaist.ir.deep.rec

import kr.ac.kaist.ir.deep.fn._
import kr.ac.kaist.ir.deep.train.Corruption

import scala.collection.mutable.ArrayBuffer

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
    var i = 0
    var last: ScalarMatrix = null
    while (i < req.size) {
      val curr = req(i)
      i += 1

      val res = curr.forward(fn)
      last = 
        if (last != null)
          last row_+ res
        else
          res
    }
    fn(last)
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
    var i = req.size - 1
    var e = err
    var seq = ArrayBuffer[TerminalNode]()
    while (i >= 0) {
      val curr = req(i)
      i -= 1
      
      val splited = spliter(e, rSize)
      val seq2 = curr.backward(splited._2, fn)
      //pass left part
      e = splited._1
      seq ++= seq2
    }
    seq
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
