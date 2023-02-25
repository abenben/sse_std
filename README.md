# SSEの標準偏差を用いたSSE改善法

SSEの標準偏差を用いたSSE改善法のサンプルコード

KMeansを使ってクラスタリングを行い、SSEの標準偏差を計算しています。クラスタ数の範囲は1からmax_kまでとなっています。データはdata.csvというファイルから読み込まれます。

エルボー点を求めるために、クラスタ数を増やした場合の期待値を計算しています。期待値を計算するためには、calc_expectation関数を使っています。この関数では、KMeansによるクラスタリングを行い、SSEを計算しています。その後、サンプル数や次元数などを用いて期待値を計算しています。

最後に、エルボー点を求めた結果をグラフに表示しています。エルボー点は、期待値の差分が最小となるクラスタ数として求められます。グラフ上では、エルボー点を示すために、破線で示しています。
