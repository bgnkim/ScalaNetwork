package kr.ac.kaist.ir.deep

import kr.ac.kaist.ir.deep.fn.ScalarMatrix
import kr.ac.kaist.ir.deep.train.Corruption

/**
 * Package object for tree
 *
 * Created by bydelta on 2015-01-14.
 */
package object rec {

  /** Tree of Vectors */
  type VectorTree = Tree[ScalarMatrix]

  /**
   * Trait: Tree
   * @tparam T is a label value for a tree
   */
  trait Tree[T] extends Serializable {
    /** Label */
    val value: T
    /** child of this tree */
    val children: Seq[Tree[T]]

    /**
     * Get child at specific index (0-based, left to right.) 
     * @param n is the index
     * @return child of given index
     */
    def apply(n: Int) = children(n)
  }

  /**
   * Tree Operation 
   * @param x to be applied
   */
  implicit class TreeOp(x: VectorTree) extends Serializable {

    /**
     * Post-Order Traversal (LC - RC - Root)
     * @param fn to be applied
     * @return final value of root node
     */
    def postOrder(fn: (ScalarMatrix, ScalarMatrix) ⇒ ScalarMatrix): ScalarMatrix =
      x match {
        case Leaf(v) ⇒ v
        case BinaryTree(v, t1, t2) ⇒
          val v1 = t1 postOrder fn
          val v2 = t2 postOrder fn
          val out = fn(v1, v2)
          v := out
      }

    /**
     * Pre-Order Traversal (Root - RC - LC)
     * @param err to be propagated
     * @param fn to be applied
     */
    def preOrder(err: ScalarMatrix, fn: ScalarMatrix ⇒ ScalarMatrix): Unit =
      x match {
        case BinaryTree(v, t1, t2) ⇒
          val e = fn(err)
          val e1 = e(0 until t1.value.rows, ::)
          val e2 = e(t1.value.rows to -1, ::)
          t2 preOrder(e2, fn)
          t1 preOrder(e1, fn)
      }

    /**
     * Pre-Order Traversal and Copy new tree (Root - RC - LC)
     * @param err to be propagated
     * @param binary to be applied to binary internal node
     * @param leaf to be applied to transformation
     */
    def preOrderCopy(err: ScalarMatrix,
                     binary: ScalarMatrix ⇒ ScalarMatrix,
                     leaf: (ScalarMatrix, ScalarMatrix) ⇒ ScalarMatrix): VectorTree =
      x match {
        case BinaryTree(v, t1, t2) ⇒
          val e = binary(err)
          val e1 = e(0 until t1.value.rows, ::)
          val e2 = e(t1.value.rows to -1, ::)
          val newt2 = t2 preOrderCopy(e2, binary, leaf)
          val newt1 = t1 preOrderCopy(e1, binary, leaf)
          BinaryTree(v, newt1, newt2)
        case Leaf(v) ⇒
          Leaf(leaf(v, err))
      }

    def ?(corrupt: Corruption): VectorTree =
      x match {
        case Leaf(v) ⇒ Leaf(corrupt(v))
        case BinaryTree(v, t1, t2) ⇒ BinaryTree(v, t1 ? corrupt, t2 ? corrupt)
      }

  }

  /**
   * Tree : Leaf node 
   * @param value for this node
   * @tparam T is a label value for a tree
   */
  case class Leaf[T](override val value: T) extends Tree[T] {
    /** Empty */
    override val children = Seq()
  }

  /**
   * Tree : Binary
   * @param value for this node
   * @param left tree
   * @param right tree
   * @tparam T is a label value for a tree
   */
  case class BinaryTree[T](override val value: T, left: Tree[T], right: Tree[T]) extends Tree[T] {
    /** Children has left and right */
    override val children = left :: right :: Nil
  }

}
