package kr.ac.kaist.ir.deep

import java.io.{ObjectInputStream, ObjectOutputStream}
import java.util

import kr.ac.kaist.ir.deep.fn._
import org.apache.log4j.Logger

import scala.collection.JavaConversions._
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
  class WordModel(val map: util.HashMap[String, ScalarMatrix]) extends Serializable with (String ⇒ ScalarMatrix) {
    lazy val vectorSize = map.head._2.rows
    private var filter: String ⇒ String = identity

    def setFilter(newFilter: String ⇒ String) = {
      filter = newFilter
    }

    /**
     * Get Matrix(Vector) of given word
     * @param word Word string for search
     * @return Column Vector of given word
     */
    def apply(word: String) = map(filter(word))

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
          val str = vec.data.map {
            v ⇒ f"${v * 10}%.0f"
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

        val buffer = new util.HashMap[String, ScalarMatrix]()
        var lineNo = mapSize

        while (lineNo > 0) {
          lineNo -= 1
          if (lineNo % 10000 == 0)
            logger info f"READ Word2Vec file : $lineNo%9d/$mapSize%9d"

          val line = br.readLine()
          val splits = line.split("\\s+")
          val word = splits(0)
          val vector = splits.view.slice(1, vectorSize + 1).map(_.toFloat).force
          require(vector.length == vectorSize, s"'$word' Vector is broken! Read size ${vector.length}, but expected $vectorSize")

          buffer += word → ScalarMatrix(vector: _*)
        }

        br.close()

        val model = new WordModel(buffer)
        val stream = new ObjectOutputStream(File(file.path + ".obj").outputStream())
        stream.writeObject(model)
        stream.close()

        logger info "READ Word2Vec finished."
        model
      }
  }
}
