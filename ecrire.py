from amr_utils.amr_readers import AMR_Reader


amrS = """# ::id PROXY_APW_ENG_20080515_0931.24 ::date 2013-07-20T23:58:49 ::snt-type body ::annotator SDL-AMR-09 ::preferred
# ::snt Yuri solomonov designed the missile and stated that the missile drops engines at a significantly lower altitude than earlier designs which makes it hard for an enemy's early warning system to detect the launch.
# ::save-date Tue Dec 19, 2017 ::file PROXY_APW_ENG_20080515_0931_24.txt
(a / and
      :op1 (d / design-01
            :ARG0 (p / person :wiki - :name (n / name :op1 "Yuri" :op2 "Solomonov"))
            :ARG1 (m / missile))
      :op2 (s / state-01
            :ARG0 p
            :ARG1 (d2 / drop-01
                  :ARG0 (m2 / missile)
                  :ARG1 (e / engine)
                  :ARG3 (a2 / altitude
                        :ARG1-of (h3 / have-degree-91
                              :ARG2 (l / low-04
                                    :ARG1 a2
                                    :ARG1-of (s3 / significant-02))
                              :ARG3 (m3 / more)
                              :ARG4 (d3 / design-01
                                    :time (b / before))))
                  :ARG0-of (m5 / make-02
                        :ARG1 (d4 / detect-01
                              :ARG0 (s2 / system
                                    :ARG0-of (w / warn-01
                                          :mod (e3 / early))
                                    :poss (p2 / person
                                          :ARG0-of (h2 / have-rel-role-91
                                                :ARG2 (e4 / enemy))))
                              :ARG1 (l2 / launch-01)
                              :ARG1-of (h / hard-02))))))
"""


reader = AMR_Reader()
amr = reader.loads(amrS, remove_wiki=True, output_alignments=False, no_tokens=True, link_string=True)
print(amr.metadata)
print("§§§§§§§§§")
print(amr.amr_string)