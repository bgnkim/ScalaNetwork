name := "DeepLearningSpark"

version := "1.0"

scalaVersion := "2.11.4"

libraryDependencies ++= Seq(
  //"org.apache.spark" %% "spark-core" % "1.2.0",
  "com.typesafe.play" %% "play-json" % "2.3.4",
  "org.scalanlp" %% "breeze" % "0.10" withJavadoc() withSources(),
  "org.scalanlp" %% "breeze-natives" % "0.10",
  "org.specs2" %% "specs2-core" % "2.4.15" % "test"
)

scalacOptions in Test ++= Seq("-Yrangepos")

resolvers ++= Seq("snapshots", "releases").map(Resolver.sonatypeRepo)