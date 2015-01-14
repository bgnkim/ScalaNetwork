package kr.ac.kaist.ir.deep

import kr.ac.kaist.ir.deep.function.ScalarMatrix

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
   * Operations for Vector Tree 
   * @param t1 to be applied
   */
  implicit class VectorTreeOp(t1: VectorTree) {
    /**
     * Construct a new binary tree 
     * @param t2 is right tree
     * @return new binary tree
     */
    def &(t2: VectorTree) = new BinaryTree[ScalarMatrix](ScalarMatrix $0(t1.value.rows, 1), t1, t2)

    /**
     * Set value of this tree 
     * @param x is the matrix to be applied
     * @return the value
     */
    def :=(x: ScalarMatrix) = {
      t1.value := x
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
