package kr.ac.kaist.ir.deep

import java.io.{ObjectInputStream, ObjectOutputStream}

import kr.ac.kaist.ir.deep.fn._
import org.apache.log4j.Logger
import org.apache.spark.AccumulatorParam

import scala.collection.mutable
import scala.io.Codec
import scala.reflect.io.{File, Path}

/**
 * Package for WordEmbedding training __(Unstable)__
 */
package object wordvec {

  /**
   * Word2Vec model class.
   * @param map Mapping between String to Array[Coord]
   */
  class WordModel(val map: Map[String, Array[Scalar]]) extends Serializable with (String ⇒ ScalarMatrix) {
    lazy val FOREIGN_VEC = ScalarMatrix(map(WordModel.FOREIGN_UNK): _*)
    lazy val NUM_VEC = ScalarMatrix(map(WordModel.NUMBERS_UNK): _*)
    lazy val OTHER_VEC = ScalarMatrix(map(WordModel.OTHER_UNK): _*)
    lazy val REAL_VEC = ScalarMatrix(map(WordModel.REALNUM): _*)
    lazy val vectorSize = OTHER_VEC.size

    /**
     * Get Matrix(Vector) of given word
     * @param word Word string for search
     * @return Column Vector of given word
     */
    def apply(word: String) =
      if (word.matches("^[0-9]+\\.[0-9]+$")) {
        REAL_VEC
      } else {
        val wordNorm =
          if (word.startsWith("#") || word.startsWith("@")) {
            //Since stanford parser preserves hashtag-like entries, remove that symbol.
            word.replaceAll("^[#|@]+", "")
          } else word

        if (map.contains(wordNorm)) {
          val entry = map(wordNorm)
          ScalarMatrix(entry: _*)
        } else if (word.matches("^[0-9]+$")) {
          NUM_VEC
        } else if (word.matches("[\u00c0-\u01ff]")) {
          FOREIGN_VEC
        } else {
          OTHER_VEC
        }
      }

    /**
     * Check whether given word mapping is UNKNOWN in Word2Vec model, or not.
     * @param word String for check
     * @return True if word is unknown.
     */
    def isUnknown(word: String) =
      if (word.matches("^[0-9]+\\.[0-9]+$")) false
      else {
        val wordNorm =
          if (word.startsWith("#") || word.startsWith("@")) {
            //Since stanford parser preserves hashtag-like entries, remove that symbol.
            word.replaceAll("^[#|@]+", "")
          } else word

        if (map.contains(wordNorm)) false
        else if (word.matches("^[0-9]+$")) false
        else true
      }

    /**
     * Write model into given path.
     * @param path Path where to store.
     */
    def saveAs(path: Path): Unit = saveAs(File(path))

    /**
     * Write model into given file.
     * @param file File where to store
     */
    def saveAs(file: File): Unit = {
      val bw = file.bufferedWriter(append = false, codec = Codec.UTF8)
      map.foreach {
        case (word, vec) ⇒
          bw.write(s"$word\t")
          val str = vec.map {
            v ⇒ f"$v%.6f"
          }.mkString(" ")
          bw.write(str)
      }
      bw.close()
    }
  }

  /**
   * Companion object of [[WordModel]]
   */
  object WordModel extends Serializable {
    final val REALNUM = "**REALNUM**"
    final val FOREIGN_UNK = "**FOREIGN**"
    final val NUMBERS_UNK = "**NUMBERS**"
    final val OTHER_UNK = "**UNKNOWN**"
    val logger = Logger.getLogger(this.getClass)

    /**
     * Restore Word Model from Path.
     * @param path Path of word model file.
     * @return WordModel restored from file.
     */
    def apply(path: Path): WordModel = apply(File(path))

    /**
     * Restore WordModel from File.
     * @param file File where to read
     * @return WordModel restored from file.
     */
    def apply(file: File): WordModel =
      if (File(file.path + ".obj").exists) {
        val in = new ObjectInputStream(File(file.path + ".obj").inputStream())
        val model = in.readObject().asInstanceOf[WordModel]
        in.close()

        logger info "READ Word2Vec finished."
        model
      } else {
        val br = file.bufferedReader(Codec.UTF8)
        val firstLine = br.readLine().split("\\s+")
        val mapSize = firstLine(0).toInt
        val vectorSize = firstLine(1).toInt

        val buffer = mutable.HashMap[String, Array[Float]]()
        var lineNo = mapSize

        while (lineNo > 0) {
          lineNo -= 1
          if (lineNo % 10000 == 0)
            logger info f"READ Word2Vec file : $lineNo%9d/$mapSize%9d"

          val line = br.readLine()
          val splits = line.split("\\s+")
          val word = splits(0)
          val vector = splits.slice(1, vectorSize + 1).map(_.toFloat)
          require(vector.length == vectorSize, s"'$word' Vector is broken! Read size ${vector.length}, but expected $vectorSize")

          buffer += word → vector
        }

        br.close()

        val model = new WordModel(buffer.toMap)
        val stream = new ObjectOutputStream(File(file.path + ".obj").outputStream())
        stream.writeObject(model)
        stream.close()

        logger info "READ Word2Vec finished."
        model
      }
  }

  /**
   * Accumulator Param object for WordEmbedding training.
   */
  implicit object WordMapAccumulator extends AccumulatorParam[mutable.HashMap[String, ScalarMatrix]] {
    /**
     * Add in place function
     * @param r1 left hand side
     * @param r2 right hand side
     * @return r1 + r2 in r1
     */
    override def addInPlace(r1: mutable.HashMap[String, ScalarMatrix],
                            r2: mutable.HashMap[String, ScalarMatrix]): mutable.HashMap[String, ScalarMatrix] = {
      r2.foreach {
        case (key, matx) if r1 contains key ⇒
          r1(key) += matx
        case (key, matx) if !r1.contains(key) ⇒
          r1.put(key, matx)
      }

      r1
    }

    /**
     * Zero value
     * @param initialValue initial value
     * @return initial zero value.
     */
    override def zero(initialValue: mutable.HashMap[String, ScalarMatrix]): mutable.HashMap[String, ScalarMatrix] =
      mutable.HashMap[String, ScalarMatrix]()
  }

}
