package kr.ac.kaist.ir.deep.wordvec

import java.io.StringReader

import edu.stanford.nlp.process.PTBTokenizer.PTBTokenizerFactory
import kr.ac.kaist.ir.deep.wordvec.WordModel
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{Logging, SparkConf, SparkContext}

import scala.collection.JavaConversions._
import scala.collection.immutable.HashSet

/**
 * Train Word2Vec and save the model.
 */
object PrepareCorpus extends Logging {
  var threshold = 5

  /**
   * Main thread.
   * @param args CLI arguments
   */
  def main(args: Array[String]) =
    if (args.length == 0 || args.contains("--help") || args.contains("-h")) {
      println(
        """Tokenize sentences, and Collect several types of unknown words.
          |
          |== Arguments without default ==
          | -i	Path of input corpora file.
          | -o	Path of tokenized output text file.
          |
          |== Arguments with default ==
          | --thre	Minimum include count. (Default: 5)
          | --part	Number of partitios. (Default: organized by Spark)
          |
          |== Additional Arguments ==
          | --help	Display this help message.
          | """.stripMargin)
    } else {
      // Set spark context
      val conf = new SparkConf()
        .setAppName("Normalize Infrequent words")
        .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .set("spark.scheduler.mode", "FAIR")
        .set("spark.shuffle.memoryFraction", "0.075")
        .set("spark.storage.unrollFraction", "0.075")
        .set("spark.storage.memoryFraction", "0.85")
      val sc = new SparkContext(conf)
      sc.setLocalProperty("spark.scheduler.pool", "production")

      val in = getArgument(args, "-i", "article.txt")
      val out = getArgument(args, "-o", "article-preproc.txt")

      threshold = getArgument(args, "--thre", "5").toInt

      // read file
      val parts = getArgument(args, "--part", "1").toInt
      val lines = sc.textFile(in, parts).filter(_.trim.nonEmpty)
      val input = getInput(lines)

      val freqWords = getWords(input.flatMap(x ⇒ x))
      val freqSet = sc.broadcast(freqWords)
      val output = getOutput(input, freqSet).cache()

      output.map(_.mkString(" ")).saveAsTextFile(out)

      // Stop the context
      sc.stop()
    }

  /**
   * Read argument
   * @param args Argument Array
   * @param key Argument Key
   * @param default Default value of this argument
   * @return Value of this key.
   */
  def getArgument(args: Array[String], key: String, default: String) = {
    val idx = args.indexOf(key)
    if (idx < 0 || idx > args.length - 1) default
    else args(idx + 1)
  }

  /**
   * Collect frequent words with count >= Threshold
   * @param words Word seq.
   * @return HashSet of frequent words.
   */
  def getWords(words: RDD[String]) = {
    val set = words.countByValue().filter(_._2 >= threshold).keySet
    HashSet.newBuilder[String].++=(set).result()
  }

  /**
   * Convert input into tokenized string, using Stanford NLP toolkit.
   * @param lines Input lines
   * @return tokenized & normalized lines.
   */
  def getInput(lines: RDD[String]) =
    lines.mapPartitions {
      @transient lazy val tokenizer = PTBTokenizerFactory.newWordTokenizerFactory("untokenizable=noneKeep")
      // Tokenize paragraph.
      // Simple split can work, but it causes the situation "ABC" != "ABC.".
      _.map {
        para ⇒
          val tokens = tokenizer.getTokenizer(new StringReader(para.trim))
            .tokenize()
          val array = new Array[String](tokens.size())

          var i = tokens.size() - 1
          while (i >= 0) {
            val value = tokens(i).value

            // Replace real numbers as "#REAL", because real number has too much diversity.
            // Natural numbers are not replaced, because they sometimes have meanings.
            array(i) =
              if (value.matches("^[0-9]+\\.[0-9]+$")) {
                WordModel.REALNUM
              } else if (value.startsWith("#") || value.startsWith("@")) {
                //Since stanford parser preserves hashtag-like entries, remove that symbol.
                value.replaceAll("^[#|@]+", "")
              } else value

            i -= 1
          }

          array
      }
    }.persist(StorageLevel.DISK_ONLY_2)

  /**
   * Convert tokenized string into a sentence, with appropriate conversion of (Threshold - 1) count word.
   * @param input Tokenized input sentence
   * @param freqSet Frequent words
   * @return Tokenized converted sentence
   */
  def getOutput(input: RDD[Array[String]], freqSet: Broadcast[HashSet[String]]) =
    input.mapPartitions {
      _.map {
        array ⇒
          var i = array.size - 1

          while (i >= 0) {
            val word = array(i)
            if (!freqSet.value.contains(word)) {
              if (word.matches("^[0-9]+$")) {
                array(i) = WordModel.NUMBERS_UNK
              } else if (word.matches("[\u00c0-\u01ff]")) {
                array(i) = WordModel.FOREIGN_UNK
              } else {
                array(i) = WordModel.OTHER_UNK
              }
            }

            i -= 1
          }

          array
      }
    }
}
