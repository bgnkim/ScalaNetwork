package kr.ac.kaist.ir.deep.rec

import kr.ac.kaist.ir.deep.fn._
import kr.ac.kaist.ir.deep.train.Corruption

import scala.collection.mutable.ArrayBuffer

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
    val iter = finals.iterator
    var last: ScalarMatrix = null
    while (iter.hasNext) {
      val curr = iter.next()
      val res = curr.forward(fn)
      last = 
        if (last != null)
          last row_+ res
        else
          res
    }
    last
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
    val iter = finals.reverseIterator
    var e = err
    var seq = ArrayBuffer[TerminalNode]()
    while (iter.hasNext) {
      val curr = iter.next()
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
  override def through(corrupt: Corruption): DAG =
    new DAG(finals map {
      _ through corrupt
    })
}