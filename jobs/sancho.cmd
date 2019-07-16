@echo off
setlocal
cd ..
:restart
call java -XX:+UseG1GC -Xmx10g -XX:MaxGCPauseMillis=800 -XX:+ExplicitGCInvokesConcurrent -Xss2m -cp bin;bin\external\reflections\reflections-0.9.9-RC1.jar;bin\external\Guava\guava-14.0.1.jar;bin\external\javassist\javassist.jar;bin\external\JUnit\junit-4.11.jar;bin\external\jna\jna-platform-4.1.0.jar;bin\external\jna\jna-4.1.0.jar;bin\external\disruptor\disruptor-3.2.1.jar;bin\external\Commons\commons-codec-1.9\commons-codec-1.9.jar;bin\external\Commons\commons-compress-1.8.1\commons-compress-1.8.1.jar;bin\external\Log4J\log4j-core-2.0-rc1.jar;bin\external\Log4J\log4j-api-2.0-rc1.jar;bin\external\Lucene\lucene-core-4.8.1.jar;bin\external\jna\jna-4.1.0.jar;bin\external\Trove\trove-3.1a1.jar;bin\external\JTidy\Tidy.jar;bin\external\Commons\commons-configuration-1.10\commons-configuration-1.10.jar;bin\external\Commons\commons-lang-2.6\commons-lang-2.6.jar;bin\external\Commons\commons-logging-1.1.3\commons-logging-1.1.3.jar;data\cfg org.ggp.base.apps.player.PlayerRunner 9147 Sancho
goto restart
