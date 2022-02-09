<TeXmacs|2.1.1>

<style|generic>

<\body>
  <doc-data|<doc-title|EM for Student's <math|t> Mixture Models>>

  <\table-of-contents|toc>
    <vspace*|1fn><with|font-series|bold|math-font-series|bold|1<space|2spc>Notation>
    <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
    <no-break><pageref|auto-1><vspace|0.5fn>

    <vspace*|1fn><with|font-series|bold|math-font-series|bold|2<space|2spc>Joint
    density> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
    <no-break><pageref|auto-2><vspace|0.5fn>

    <vspace*|1fn><with|font-series|bold|math-font-series|bold|3<space|2spc>Complete
    data log-likelihood> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
    <no-break><pageref|auto-3><vspace|0.5fn>

    <vspace*|1fn><with|font-series|bold|math-font-series|bold|4<space|2spc>Expected
    data log-likelihood> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
    <no-break><pageref|auto-4><vspace|0.5fn>

    <with|par-left|1tab|4.1<space|2spc>Objective for each parameter group
    <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
    <no-break><pageref|auto-5>>

    <with|par-left|1tab|4.2<space|2spc>E-step: evaluating expectations
    <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
    <no-break><pageref|auto-6>>

    <with|par-left|1tab|4.3<space|2spc>M-step: solving the objectives
    <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
    <no-break><pageref|auto-7>>
  </table-of-contents>

  <section|Notation>

  You will see me being too careful about notation in this derivation. This
  is because this has not been an easy derivation and being explicit actually
  helps.

  <\itemize>
    <item>Let <math|y<rsub|j>> be the observed values of the <math|j>-th data
    vector (so <math|y<rsub|j>> is a vector, and I'm not using vector
    notation to save up some typing).

    <item>Let <math|z<rsub|j>> be the <with|font-shape|italic|integer> value
    of the corresponding mixture component index.

    <item>Let <math|u<rsub|j>> be the <with|font-shape|italic|scalar> value
    of the corresponding latent variable in the Gaussian scale mixture.

    <item>Let <math|g> denote the number of mixture components and <math|n>
    denote the number of training examples.

    <item>Let <math|\<theta\>> denote all parameters, i.e.,
    <math|\<theta\>=<around*|{|\<pi\>,<with|font-series|bold|\<mu\>><rsub|1:g>,<with|font-series|bold|\<Sigma\>><rsub|1:g>,v<rsub|1:g>|}>>.
  </itemize>

  Finally, the bulk of this derivation is self-contained and, for the most
  part, you do not need to refer to the original paper unless you point you
  there. The paper also contains some errors here and there so be careful.
  Please send me an email if you spot an error in this derivation.

  <section|Joint density>

  To prepare for Section <reference|complete_data_ll>, we want to express
  <math|p<rsub|Z<rsub|j>,U<rsub|j>,Y<rsub|j>><around|(|z<rsub|j>,u<rsub|j>,y<rsub|j>\<mid\>\<theta\>|)>>
  in distributions that we already know (i.e., as defined by the forward
  model):

  <\equation*>
    p<rsub|Z<rsub|j>,U<rsub|j>,Y<rsub|j>><around|(|z<rsub|j>,u<rsub|j>,y<rsub|j>;\<theta\>|)>=p<rsub|Y<rsub|j>\<mid\>Z<rsub|j>,U<rsub|j>><around|(|y<rsub|j>\<mid\>z<rsub|j>,u<rsub|j>;\<theta\>|)>p<rsub|U<rsub|j>\<mid\>Z<rsub|j>><around|(|u<rsub|j>\<mid\>z<rsub|j>;\<theta\>|)>p<rsub|Z<rsub|j>><around|(|z<rsub|j>;\<theta\>|)>
  </equation*>

  where:

  <\eqnarray*>
    <tformat|<table|<row|<cell|p<rsub|Z<rsub|j>><around|(|z<rsub|j>;\<theta\>|)>>|<cell|=>|<cell|<big|prod><rsub|i=1><rsup|g>\<pi\><rsub|i><rsup|\<bbb-I\><around|(|z<rsub|j>=i|)>>>>|<row|<cell|p<rsub|U<rsub|j>\<mid\>Z<rsub|j>><around|(|u<rsub|j>\<mid\>z<rsub|j>;\<theta\>|)>>|<cell|=>|<cell|Ga<around|(|u<rsub|j>\<mid\><frac|v<rsub|z<rsub|j>>|2>,<frac|v<rsub|z<rsub|j>>|2>|)>>>|<row|<cell|>|<cell|=>|<cell|<big|prod><rsub|i=1><rsup|g>Ga<around|(|u<rsub|j>\<mid\><frac|v<rsub|i>|2>,<frac|v<rsub|i>|2>|)><rsup|\<bbb-I\><around|(|z<rsub|j>=i|)>>>>|<row|<cell|p<rsub|Y<rsub|j>\<mid\>U<rsub|j>,Z<rsub|j>><around|(|y<rsub|j>\<mid\>u<rsub|j>,z<rsub|j>;\<theta\>|)>>|<cell|=>|<cell|<with|font|cal|N><around|(|y<rsub|j>\<mid\><with|font-series|bold|\<mu\>><rsub|z<rsub|j>>,<with|font-series|bold|\<Sigma\>><rsub|z<rsub|j>>/u<rsub|j>|)>>>|<row|<cell|>|<cell|=>|<cell|<big|prod><rsub|i=1><rsup|g><with|font|cal|N><around|(|y<rsub|j>\<mid\><with|font-series|bold|\<mu\>><rsub|i><rsub|>,<with|font-series|bold|\<Sigma\>><rsub|i>/u<rsub|j>|)><rsup|\<bbb-I\><around|(|z<rsub|j>=i|)>>>>>>
  </eqnarray*>

  The above notation of multiplying <math|g> terms of which <math|g-1> terms
  is just 1 is a notation used to simplify later calculations.

  <section|Complete data log-likelihood><label|complete_data_ll>

  For the complete data log likelihood (we'll take expectation later in
  Section <reference|expected_data_ll>), this joint distribution factors and
  we get

  <\equation*>
    <big|sum><rsub|j=1><rsup|n>log p<rsub|Z<rsub|j>,U<rsub|j>,Y<rsub|j>><around|(|z<rsub|j>,u<rsub|j>,y<rsub|j>;\<theta\>|)>=<big|sum><rsub|j=1><rsup|n>log
    p<rsub|Y<rsub|j>\<mid\>Z<rsub|j>,U<rsub|j>><around|(|y<rsub|j>\<mid\>z<rsub|j>,u<rsub|j>;\<theta\>|)>+<big|sum><rsub|j=1><rsup|n>log
    p<rsub|U<rsub|j>\<mid\>Z<rsub|j>><around|(|u<rsub|j>\<mid\>z<rsub|j>;\<theta\>|)>+<big|sum><rsub|j=1><rsup|n>log
    p<rsub|Z<rsub|j>><around|(|z<rsub|j>;\<theta\>|)>
  </equation*>

  where each term is concerned about different parameters (which is great
  isn't it?):

  <\eqnarray*>
    <tformat|<table|<row|<cell|<big|sum><rsub|j=1><rsup|n>log
    p<rsub|Z<rsub|j>><around|(|z<rsub|j>;\<theta\>|)>>|<cell|=>|<cell|<big|sum><rsub|j=1><rsup|n>log
    <big|prod><rsub|i=1><rsup|g>\<pi\><rsub|i><rsup|\<bbb-I\><around|(|z<rsub|j>=i|)>>>>|<row|<cell|>|<cell|=>|<cell|
    <big|sum><rsub|i=1><rsup|g><big|sum><rsub|j=1><rsup|n>\<bbb-I\><around|(|z<rsub|j>=i|)>
    log \<pi\><rsub|i>>>|<row|<cell|>|<cell|>|<cell|>>|<row|<cell|<big|sum><rsub|j=1><rsup|n>log
    p<rsub|U<rsub|j>\<mid\>Z<rsub|j>><around|(|u<rsub|j>\<mid\>z<rsub|j>;\<theta\>|)>>|<cell|=>|<cell|<big|sum><rsub|j=1><rsup|n>log
    <big|prod><rsub|i=1><rsup|g>Ga<around|(|u<rsub|j>\<mid\><frac|v<rsub|i>|2>,<frac|v<rsub|i>|2>|)><rsup|\<bbb-I\><around|(|z<rsub|j>=i|)>>>>|<row|<cell|>|<cell|=>|<cell|<big|sum><rsub|i=1><rsup|g><big|sum><rsub|j=1><rsup|n>\<bbb-I\><around|(|z<rsub|j>=i|)>
    log Ga<around|(|u<rsub|j>\<mid\><frac|v<rsub|i>|2>,<frac|v<rsub|i>|2>|)>>>|<row|<cell|>|<cell|=>|<cell|<big|sum><rsub|i=1><rsup|g><big|sum><rsub|j=1><rsup|n>\<bbb-I\><around|(|z<rsub|j>=i|)>
    log <around|<left|[|3>|<frac|<around|(|v<rsub|i>/2|)><rsup|v<rsub|i>/2>|\<Gamma\><around|(|v<rsub|i>/2|)>>\<cdot\>u<rsub|j><rsup|v<rsub|i>/2-1>\<cdot\>exp<around|{|-u<rsub|j>\<cdot\><around|(|v<rsub|i>/2|)>|}>|<right|]|3>>>>|<row|<cell|>|<cell|=>|<cell|<big|sum><rsub|i=1><rsup|g><big|sum><rsub|j=1><rsup|n>\<bbb-I\><around|(|z<rsub|j>=i|)>
    <around|<left|(|2>|<frac|v<rsub|i>|2> log <frac|v<rsub|i>|2>-log
    \<Gamma\><around|(|v<rsub|i>/2|)>+<around|<left|(|2>|<frac|v<rsub|i>|2>-1|<right|)|2>>
    log u<rsub|j>-<frac|v<rsub|i>|2>\<cdot\>u<rsub|j>|<right|)|2>>>>|<row|<cell|>|<cell|=>|<cell|<big|sum><rsub|i=1><rsup|g><big|sum><rsub|j=1><rsup|n>\<bbb-I\><around|(|z<rsub|j>=i|)>
    <around|<left|(|2>|<frac|v<rsub|i>|2> log <frac|v<rsub|i>|2>-log
    \<Gamma\><around|(|v<rsub|i>/2|)>+<frac|v<rsub|i>|2> <around|(|log
    u<rsub|j>-u<rsub|j>|)>-log u<rsub|j>|<right|)|2>>>>|<row|<cell|>|<cell|=<rsub|grad>>|<cell|<big|sum><rsub|i=1><rsup|g><big|sum><rsub|j=1><rsup|n>\<bbb-I\><around|(|z<rsub|j>=i|)>
    <around|<left|(|2>|<frac|v<rsub|i>|2> log <frac|v<rsub|i>|2>-log
    \<Gamma\><around|(|v<rsub|i>/2|)>+<frac|v<rsub|i>|2> <around|(|log
    u<rsub|j>-u<rsub|j>|)>|<right|)|2>>>>|<row|<cell|>|<cell|>|<cell|>>|<row|<cell|<big|sum><rsub|j=1><rsup|n>log
    p<rsub|Y<rsub|j>\<mid\>Z<rsub|j>,U<rsub|j>><around|(|y<rsub|j>\<mid\>z<rsub|j>,u<rsub|j>;\<theta\>|)>>|<cell|=>|<cell|<big|sum><rsub|j=1><rsup|n>log
    <big|prod><rsub|i=1><rsup|g><with|font|cal|N><around|(|y<rsub|j>\<mid\><with|font-series|bold|\<mu\>><rsub|i><rsub|>,<with|font-series|bold|\<Sigma\>><rsub|i>/u<rsub|j>|)><rsup|\<bbb-I\><around|(|z<rsub|j>=i|)>>>>|<row|<cell|>|<cell|=>|<cell|<big|sum><rsub|i=1><rsup|g><big|sum><rsub|j=1><rsup|n>\<bbb-I\><around|(|z<rsub|j>=i|)>
    log <with|font|cal|N><around|(|y<rsub|j>\<mid\><with|font-series|bold|\<mu\>><rsub|i><rsub|>,<with|font-series|bold|\<Sigma\>><rsub|i>/u<rsub|j>|)>>>|<row|<cell|>|<cell|=>|<cell|<big|sum><rsub|i=1><rsup|g><big|sum><rsub|j=1><rsup|n>\<bbb-I\><around|(|z<rsub|j>=i|)>
    log <around|<left|[|3>|<frac|1|<around|(|2\<pi\>|)><rsup|p/2><around|\||<with|font-series|bold|\<Sigma\>><rsub|i>/u<rsub|j>|\|><rsup|1/2>>
    exp<around|<left|{|3>|-<frac|1|2><around|(|y<rsub|j>-<with|font-series|bold|\<mu\>><rsub|j>|)><rsup|T><around|(|<with|font-series|bold|\<Sigma\>><rsub|i><rsup|>/u<rsub|j>|)><rsup|-1><around|(|y<rsub|j>-<with|font-series|bold|\<mu\>><rsub|j>|)>|<right|}|3>>|<right|]|3>>>>|<row|<cell|>|<cell|=>|<cell|<big|sum><rsub|i=1><rsup|g><big|sum><rsub|j=1><rsup|n>\<bbb-I\><around|(|z<rsub|j>=i|)>
    log <around|<left|[|3>|<frac|u<rsub|j><rsup|1/2>|<around|(|2\<pi\>|)><rsup|p/2><around|\||<with|font-series|bold|\<Sigma\>><rsub|i>|\|><rsup|1/2>>
    exp<around|<left|{|3>|-<frac|u<rsub|j>|2><around|(|y<rsub|j>-<with|font-series|bold|\<mu\>><rsub|j>|)><rsup|T><with|font-series|bold|\<Sigma\>><rsub|i><rsup|-1><around|(|y<rsub|j>-<with|font-series|bold|\<mu\>><rsub|j>|)>|<right|}|3>>|<right|]|3>>>>|<row|<cell|>|<cell|=>|<cell|<big|sum><rsub|i=1><rsup|g><big|sum><rsub|j=1><rsup|n>\<bbb-I\><around|(|z<rsub|j>=i|)>
    <around|<left|(|2>|-<frac|p|2> log <around|(|2\<pi\>|)>-<frac|1|2>log
    <around|\||<with|font-series|bold|\<Sigma\>><rsub|i>|\|>+<frac|1|2>log
    u<rsub|j>-<frac|u<rsub|j>|2><around|(|y<rsub|j>-<with|font-series|bold|\<mu\>><rsub|j>|)><rsup|T><with|font-series|bold|\<Sigma\>><rsub|i><rsup|-1><around|(|y<rsub|j>-<with|font-series|bold|\<mu\>><rsub|j>|)>|<right|)|2>>>>|<row|<cell|>|<cell|=<rsub|<text|grad>>>|<cell|<big|sum><rsub|i=1><rsup|g><big|sum><rsub|j=1><rsup|n>\<bbb-I\><around|(|z<rsub|j>=i|)>
    <around|<left|(|2>|-<frac|1|2>log <around|\||<with|font-series|bold|\<Sigma\>><rsub|i>|\|>-<frac|u<rsub|j>|2><around|(|y<rsub|j>-<with|font-series|bold|\<mu\>><rsub|j>|)><rsup|T><with|font-series|bold|\<Sigma\>><rsub|i><rsup|-1><around|(|y<rsub|j>-<with|font-series|bold|\<mu\>><rsub|j>|)>|<right|)|2>>>>>>
  </eqnarray*>

  <section|Expected data log-likelihood>

  <subsection|Objective for each parameter
  group><label|obj><label|expected_data_ll>

  Now comes the time to compute the expected log likelihood over
  <math|Z<rsub|j>>'s and <math|U<rsub|j>>'s. We don't have to take this
  expectation over these random variables together in one expectation.
  Instead, we can break it into an inner and an outer expectation and take
  the two sequentially. Conveniently, the expectation will by linearity
  factor across the three terms so we can compute the expectation for one
  term at a time:

  <with|font-series|bold|First term >(objective for for <math|\<pi\>>)

  <\eqnarray*>
    <tformat|<table|<row|<cell|>|<cell|>|<cell|\<bbb-E\><rsub|Z<rsub|j>\<sim\>p<rsub|Z<rsub|j>\<mid\>Y<rsub|j>><around|(|\<cdot\>\<mid\>y<rsub|j>;\<theta\><rsub|old>|)>><around|<left|[|4>|<big|sum><rsub|j=1><rsup|n>log
    p<rsub|Z<rsub|j>><around|(|Z<rsub|j>;\<theta\>|)>|<right|]|4>>>>|<row|<cell|>|<cell|=>|<cell|\<bbb-E\><rsub|Z<rsub|j>\<sim\>p<rsub|Z<rsub|j>\<mid\>Y<rsub|j>><around|(|\<cdot\>\<mid\>y<rsub|j>;\<theta\><rsub|old>|)>><around|<left|[|4>|<big|sum><rsub|i=1><rsup|g><big|sum><rsub|j=1><rsup|n>\<bbb-I\><around|(|Z<rsub|j>=i|)>
    log \<pi\><rsub|i>|<right|]|4>>>>|<row|<cell|>|<cell|=>|<cell|<big|sum><rsub|i=1><rsup|g><big|sum><rsub|j=1><rsup|n>\<bbb-E\><rsub|Z<rsub|j>\<sim\>p<rsub|Z<rsub|j>\<mid\>Y<rsub|j>><around|(|\<cdot\>\<mid\>y<rsub|j>;\<theta\><rsub|old>|)>><around|[|\<bbb-I\><around|(|Z<rsub|j>=i|)>|]>
    log \<pi\><rsub|i>>>|<row|<cell|>|<cell|=>|<cell|<big|sum><rsub|i=1><rsup|g><big|sum><rsub|j=1><rsup|n><wide*|p<rsub|Z<rsub|j>\<mid\>Y<rsub|j>><around|(|i\<mid\>y<rsub|j>;\<theta\><rsub|old>|)>|\<wide-underbrace\>><rsub|\<tau\><rsub|i
    j>> log \<pi\><rsub|i>>>>>
  </eqnarray*>

  <with|font-series|bold|Second term> (objective for <math|v<rsub|i>>'s)

  <\eqnarray*>
    <tformat|<cwith|1|-1|2|-1|font-base-size|8>|<table|<row|<cell|>|<cell|>|<cell|\<bbb-E\><rsub|Z<rsub|j>,U<rsub|j>\<sim\>p<rsub|Z<rsub|j>,U<rsub|j>\<mid\>Y<rsub|j>><around|(|\<cdot\>,\<cdot\>\<mid\>y<rsub|j>;\<theta\><rsub|old>|)>><around|<left|[|4>|<big|sum><rsub|j=1><rsup|n>log
    p<rsub|U<rsub|j>\<mid\>Z<rsub|j>><around|(|U<rsub|j>\<mid\>Z<rsub|j>,\<theta\>|)>|<right|]|4>>>>|<row|<cell|>|<cell|=>|<cell|\<bbb-E\><rsub|Z<rsub|j>\<sim\>p<rsub|Z<rsub|j>\<mid\>Y<rsub|j>><around|(|\<cdot\>\<mid\>y<rsub|j>;\<theta\><rsub|old>|)>><around|<left|[|4>|\<bbb-E\><rsub|U<rsub|j>\<sim\>p<rsub|U<rsub|j>\<mid\>Z<rsub|j>,Y<rsub|j><around|(|\<cdot\>\<mid\>Z<rsub|j>,y<rsub|j>;\<theta\><rsub|old>|)>>><around|<left|[|4>|<big|sum><rsub|i=1><rsup|g><big|sum><rsub|j=1><rsup|n>\<bbb-I\><around|(|Z<rsub|j>=i|)>
    <around|<left|(|2>|<frac|v<rsub|i>|2> log <frac|v<rsub|i>|2>-log
    \<Gamma\><around|(|v<rsub|i>/2|)>+<frac|v<rsub|i>|2> <around|(|log
    U<rsub|j>-U<rsub|j>|)>-log U<rsub|j>|<right|)|2>>|<right|]|4>>|<right|]|4>>>>|<row|<cell|>|<cell|=>|<cell|<with|font-base-size|8|\<bbb-E\><rsub|Z<rsub|j>\<sim\>p<rsub|Z<rsub|j>\<mid\>Y<rsub|j>><around|(|\<cdot\>\<mid\>y<rsub|j>;\<theta\><rsub|old>|)>><around*|[|<big|sum><rsub|i=1><rsup|g><big|sum><rsub|j=1><rsup|n>\<bbb-I\><around|(|Z<rsub|j>=i|)>
    <around*|(|<frac|v<rsub|i>|2> log <frac|v<rsub|i>|2>-log
    \<Gamma\><around|(|v<rsub|i>/2|)>+<frac|v<rsub|i>|2>
    <around|(|<around|\<nobracket\>|\<bbb-E\><rsub|U<rsub|j>\<sim\>p<rsub|U<rsub|j>\<mid\>Z<rsub|j>,Y<rsub|j>><around|(|\<cdot\>\<mid\>Z<rsub|j>,y<rsub|j>;\<theta\><rsub|old>|)>><around|[|log
    U<rsub|j>|]>-\<bbb-E\><rsub|U<rsub|j>\<sim\>p<rsub|U<rsub|j>\<mid\>Z<rsub|j>,Y<rsub|j>><around|(|\<cdot\>\<mid\>Z<rsub|j>,y<rsub|j>;\<theta\><rsub|old>|)>><around|[||\<nobracket\>>U<rsub|j>|]>|)>
    <rsub|>-\<bbb-E\><rsub|U<rsub|j>\<sim\>p<rsub|U<rsub|j>\<mid\>Z<rsub|j>,Y<rsub|j>><around|(|\<cdot\>\<mid\>Z<rsub|j>,y<rsub|j>;\<theta\><rsub|old>|)>><around|[|log
    U<rsub|j>|]>|)>|]>>>>|<row|<cell|>|<cell|=>|<cell|<with|font-base-size|8|\<bbb-E\><rsub|Z<rsub|j>\<sim\>p<rsub|Z<rsub|j>\<mid\>Y<rsub|j>><around|(|\<cdot\>\<mid\>y<rsub|j>;\<theta\><rsub|old>|)>><around*|[|<big|sum><rsub|i=1><rsup|g><big|sum><rsub|j=1><rsup|n>\<bbb-I\><around|(|Z<rsub|j>=i|)>
    <around*|(|<frac|v<rsub|i>|2> log <frac|v<rsub|i>|2>-log
    \<Gamma\><around|(|v<rsub|i>/2|)>+<frac|v<rsub|i>|2>
    <around|<left|(|1>|<around|\<nobracket\>|\<bbb-E\><rsub|U<rsub|j>\<sim\>p<rsub|U<rsub|j>\<mid\>Z<rsub|j>,Y<rsub|j>><around|(|\<cdot\>\<mid\>1,y<rsub|j>;\<theta\><rsub|old>|)>><around|[|log
    U<rsub|j>|]>-\<bbb-E\><rsub|u<rsub|j>\<sim\>p<rsub|U<rsub|j>\<mid\>Z<rsub|j>,Y<rsub|j>><around|(|\<cdot\>\<mid\>1,y<rsub|j>;\<theta\><rsub|old>|)>><around|[||\<nobracket\>>Y<rsub|j>|]>|<right|)|1>>-\<bbb-E\><rsub|U<rsub|j>\<sim\>p<rsub|U<rsub|j>\<mid\>Z<rsub|j>,Y<rsub|j>><around|(|\<cdot\>\<mid\>1,y<rsub|j>;\<theta\><rsub|old>|)>><around|[|log
    U<rsub|j>|]>|)>|]>><space|1em><around|(|\<ast\>|)>>>|<row|<cell|>|<cell|=>|<cell|<big|sum><rsub|i=1><rsup|g><big|sum><rsub|j=1><rsup|n>\<bbb-E\><rsub|Z<rsub|j>\<sim\>p<rsub|Z<rsub|j>\<mid\>Y<rsub|j>><around|(|\<cdot\>\<mid\>y<rsub|j>;\<theta\><rsub|old>|)>><around|[|\<bbb-I\><around|(|Z<rsub|j>=i|)>|]>
    <around*|(|<frac|v<rsub|i>|2> log <frac|v<rsub|i>|2>-log
    \<Gamma\><around|(|v<rsub|i>/2|)>+<frac|v<rsub|i>|2>
    <around|<left|(|1>|<around|\<nobracket\>|\<bbb-E\><rsub|U<rsub|j>\<sim\>p<rsub|U<rsub|j>\<mid\>Z<rsub|j>,Y<rsub|j>><around|(|\<cdot\>\<mid\>1,y<rsub|j>;\<theta\><rsub|old>|)>><around|[|log
    U<rsub|j>|]>-\<bbb-E\><rsub|U<rsub|j>\<sim\>p<rsub|U<rsub|j>\<mid\>Z<rsub|j>,Y<rsub|j>><around|(|\<cdot\>\<mid\>1,y<rsub|j>;\<theta\><rsub|old>|)>><around|[||\<nobracket\>>U<rsub|j>|]>|<right|)|1>>-\<bbb-E\><rsub|u<rsub|j>\<sim\>p<rsub|U<rsub|j>\<mid\>Z<rsub|j>,Y<rsub|j>><around|(|u<rsub|j>\<mid\>1,y<rsub|j>;\<theta\><rsub|old>|)>><around|[|log
    U<rsub|j>|]>|)>>>|<row|<cell|>|<cell|=>|<cell|<big|sum><rsub|i=1><rsup|g><big|sum><rsub|j=1><rsup|n><wide*|p<rsub|Z<rsub|j>\<mid\>Y<rsub|j>><around|(|i\<mid\>y<rsub|j>;\<theta\><rsub|old>|)>|\<wide-underbrace\>><rsub|\<tau\><rsub|i
    j>> <around*|<left|(|-1>|<frac|v<rsub|i>|2> log <frac|v<rsub|i>|2>-log
    \<Gamma\><around|(|v<rsub|i>/2|)>+<frac|v<rsub|i>|2>
    <around|<left|(|1>|<wide*|\<bbb-E\><rsub|U<rsub|j>\<sim\>p<rsub|U<rsub|j>\<mid\>Z<rsub|j>,Y<rsub|j>><around|(|\<cdot\>\<mid\>1,y<rsub|j>;\<theta\><rsub|old>|)>><around|[|log
    U<rsub|j>|]>|\<wide-underbrace\>><rsub|<text|see
    paper>>-<wide*|\<bbb-E\><rsub|U<rsub|j>\<sim\>p<rsub|U<rsub|j>\<mid\>Z<rsub|j>,Y<rsub|j>><around|(|\<cdot\>\<mid\>1,y<rsub|j>;\<theta\><rsub|old>|)>><around|[|U<rsub|j>|]>|\<wide-underbrace\>><rsub|<text|see
    paper>>|<right|)|1>>-\<bbb-E\><rsub|U<rsub|j>\<sim\>p<rsub|U<rsub|j>\<mid\>Z<rsub|j>,Y<rsub|j>><around|(|\<cdot\>\<mid\>1,y<rsub|j>;\<theta\><rsub|old>|)>><around|[|log
    U<rsub|j>|]>|<right|)|-1>>>>>>
  </eqnarray*>

  <with|font-series|bold|Third term> (objective for
  <math|<with|font-series|bold|\<mu\>><rsub|i>>'s and
  <math|<with|font-series|bold|\<Sigma\>><rsub|i>>'s)

  <\eqnarray*>
    <tformat|<table|<row|<cell|>|<cell|>|<cell|\<bbb-E\><rsub|Z<rsub|j>,U<rsub|j>\<sim\>p<rsub|Z<rsub|j>,U<rsub|j>\<mid\>Y<rsub|j>><around|(|\<cdot\>,\<cdot\>\<mid\>y<rsub|j>;\<theta\><rsub|old>|)>><around|<left|[|4>|<big|sum><rsub|j=1><rsup|n>log
    p<rsub|Y<rsub|j>\<mid\>Z<rsub|j>,U<rsub|j>><around|(|y<rsub|j>\<mid\>Z<rsub|j>,U<rsub|j>;\<theta\>|)>|<right|]|4>>>>|<row|<cell|>|<cell|=>|<cell|\<bbb-E\><rsub|Z<rsub|j>\<sim\>p<rsub|Z<rsub|j>\<mid\>Y<rsub|j>><around|(|\<cdot\>\<mid\>y<rsub|j>;\<theta\><rsub|old>|)>><around|<left|[|4>|\<bbb-E\><rsub|U<rsub|j>\<sim\>p<rsub|U<rsub|j>\<mid\>Z<rsub|j>,Y<rsub|j>><around|(|\<cdot\>\<mid\>Z<rsub|j>,y<rsub|j>;\<theta\><rsub|old>|)>><around|<left|[|4>|<big|sum><rsub|j=1><rsup|n>log
    p<rsub|Y<rsub|j>\<mid\>Z<rsub|j>,U<rsub|j>><around|(|y<rsub|j>\<mid\>Z<rsub|j>,U<rsub|j>;\<theta\>|)>|<right|]|4>>|<right|]|4>>>>|<row|<cell|>|<cell|=>|<cell|\<bbb-E\><rsub|Z<rsub|j>\<sim\>p<rsub|Z<rsub|j>\<mid\>Y<rsub|j>><around|(|\<cdot\>\<mid\>y<rsub|j>;\<theta\><rsub|old>|)>><around*|<left|[|-1>|\<bbb-E\><rsub|U<rsub|j>\<sim\>p<rsub|U<rsub|j>\<mid\>Z<rsub|j>,Y<rsub|j>><around|(|\<cdot\>\<mid\>Z<rsub|j>,y<rsub|j>;\<theta\><rsub|old>|)>><around*|<left|[|-1>|<big|sum><rsub|i=1><rsup|g><big|sum><rsub|j=1><rsup|n>\<bbb-I\><around|(|Z<rsub|j>=i|)>
    <around|<left|(|3>|-<frac|1|2>log <around|\||<with|font-series|bold|\<Sigma\>><rsub|i>|\|>-<frac|U<rsub|j>|2><around|(|y<rsub|j>-<with|font-series|bold|\<mu\>><rsub|j>|)><rsup|T><with|font-series|bold|\<Sigma\>><rsub|i><rsup|-1><around|(|y<rsub|j>-<with|font-series|bold|\<mu\>><rsub|j>|)>|<right|)|3>>|<right|]|-1>>|<right|]|-1>>>>|<row|<cell|>|<cell|=>|<cell|\<bbb-E\><rsub|Z<rsub|j>\<sim\>p<rsub|Z<rsub|j>\<mid\>Y<rsub|j>><around|(|\<cdot\>\<mid\>y<rsub|j>;\<theta\><rsub|old>|)>><around*|<left|[|-1>|<big|sum><rsub|i=1><rsup|g><big|sum><rsub|j=1><rsup|n>\<bbb-I\><around|(|Z<rsub|j>=i|)>
    <around|<left|(|3>|-<frac|1|2>log <around|\||<with|font-series|bold|\<Sigma\>><rsub|i>|\|>-<frac|1|2>\<bbb-E\><rsub|U<rsub|j>\<sim\>p<rsub|U<rsub|j>\<mid\>Y<rsub|j>,Z<rsub|j>><around|(|\<cdot\>\<mid\>y<rsub|j>,Z<rsub|j>;\<theta\>|)>><around|[|U<rsub|j>|]><around|(|y<rsub|j>-<with|font-series|bold|\<mu\>><rsub|j>|)><rsup|T><with|font-series|bold|\<Sigma\>><rsub|i><rsup|-1><around|(|y<rsub|j>-<with|font-series|bold|\<mu\>><rsub|j>|)>|<right|)|3>>|<right|]|-1>>>>|<row|<cell|>|<cell|=>|<cell|\<bbb-E\><rsub|Z<rsub|j>\<sim\>p<rsub|Z<rsub|j>\<mid\>Y<rsub|j>><around|(|\<cdot\>\<mid\>y<rsub|j>;\<theta\><rsub|old>|)>><around*|<left|[|-1>|<big|sum><rsub|i=1><rsup|g><big|sum><rsub|j=1><rsup|n>\<bbb-I\><around|(|Z<rsub|j>=i|)>
    <around|<left|(|3>|-<frac|1|2>log <around|\||<with|font-series|bold|\<Sigma\>><rsub|i>|\|>-<frac|1|2>\<bbb-E\><rsub|U<rsub|j>\<sim\>p<rsub|U<rsub|j>\<mid\>Y<rsub|j>,Z<rsub|j>><around|(|\<cdot\>\<mid\>y<rsub|j>,1;\<theta\>|)>><around|[|U<rsub|j>|]><around|(|y<rsub|j>-<with|font-series|bold|\<mu\>><rsub|j>|)><rsup|T><with|font-series|bold|\<Sigma\>><rsub|i><rsup|-1><around|(|y<rsub|j>-<with|font-series|bold|\<mu\>><rsub|j>|)>|<right|)|3>>|<right|]|-1>><space|1em><around|(|\<ast\>|)>>>|<row|<cell|>|<cell|=>|<cell|<big|sum><rsub|i=1><rsup|g><big|sum><rsub|j=1><rsup|n><wide*|\<bbb-E\><rsub|Z<rsub|j>\<sim\>p<rsub|Z<rsub|j>\<mid\>Y<rsub|j>><around|(|\<cdot\>\<mid\>y<rsub|j>;\<theta\><rsub|old>|)>><around|[|\<bbb-I\><around|(|Z<rsub|j>=i|)>|]>|\<wide-underbrace\>><rsub|\<tau\><rsub|i
    j>> <around|<left|(|3>|-<frac|1|2>log
    <around|\||<with|font-series|bold|\<Sigma\>><rsub|i>|\|>-<frac|1|2><wide*|\<bbb-E\><rsub|U<rsub|j>\<sim\>p<rsub|U<rsub|j>\<mid\>Y<rsub|j>,Z<rsub|j>><around|(|\<cdot\>\<mid\>y<rsub|j>,1;\<theta\>|)>><around|[|U<rsub|j>|]>|\<wide-underbrace\>><around|(|y<rsub|j>-<with|font-series|bold|\<mu\>><rsub|j>|)><rsup|T><with|font-series|bold|\<Sigma\>><rsub|i><rsup|-1><around|(|y<rsub|j>-<with|font-series|bold|\<mu\>><rsub|j>|)>|<right|)|3>>>>>>
  </eqnarray*>

  (\<ast\>) The term behind <math|\<bbb-I\><around*|(|Z<rsub|j>=1|)>> is
  taken into account (i.e., not zero) only when <math|Z<rsub|j>=1>, so we can
  safely set <math|Z<rsub|j>=1> in that term. By doing so, that term turns
  into a constant rather a random variable. <with|font-shape|italic|I believe
  this is the most important step in the whole derivation; neither the
  original paper nor Murphy's book talk about this step, and I was stuck
  there for some time.>

  <subsection|E-step: evaluating expectations>

  In the E-step, we compute the stuff enclosed in curly brackets in last
  subsection. Please refer to Section 6 of the original paper for this stuff
  \U that section is easy to follow and error-free.

  <subsection|M-step: solving the objectives>

  In the M-step, we solve for the maxima of the objectives in Section
  <reference|obj>. Please refer to Section 7 of the original paper for this
  stuff \U that section is easy to follow but contains an error in Equation
  32, which I've commented about in my code.
</body>

<\initial>
  <\collection>
    <associate|page-medium|papyrus>
    <associate|page-orientation|landscape>
    <associate|page-screen-margin|false>
  </collection>
</initial>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|1>>
    <associate|auto-2|<tuple|2|1>>
    <associate|auto-3|<tuple|3|1>>
    <associate|auto-4|<tuple|4|1>>
    <associate|auto-5|<tuple|4.1|?>>
    <associate|auto-6|<tuple|4.2|?>>
    <associate|auto-7|<tuple|4.3|?>>
    <associate|complete_data_ll|<tuple|3|?>>
    <associate|expected_data_ll|<tuple|4.1|?>>
    <associate|obj|<tuple|4.1|?>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|toc>
      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|1<space|2spc>Notation>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|2<space|2spc>Joint
      density> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-2><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|3<space|2spc>Complete
      data log-likelihood> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-3><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|4<space|2spc>Expected
      data log-likelihood> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-4><vspace|0.5fn>

      <with|par-left|<quote|1tab>|4.1<space|2spc>Objective for each parameter
      group <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-5>>

      <with|par-left|<quote|1tab>|4.2<space|2spc>E-step: evaluating
      expectations <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-6>>

      <with|par-left|<quote|1tab>|4.3<space|2spc>M-step: solving the
      objectives <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-7>>
    </associate>
  </collection>
</auxiliary>