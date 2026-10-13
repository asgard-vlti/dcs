#!/usr/bin/env python3
"""Single-file Baldr simulator calibration.

Copy this script and settings.simulation.json to the destination directory.
Run: python simulate_standalone.py --output runs/trial1
Use --offline for the synthetic instrument; --objective rmse for RMSE priority.
Default completion restores the starting DM state; --apply-best leaves the result.

Requires numpy/scipy; connected mode also xaosim/pyzmq; --plot needs matplotlib.
The default optical configuration and material table are included below. Explicit
custom data paths in settings still require those files. This is a snapshot of
2026-10-14; changes to the multi-file package do not automatically update it.
"""
from __future__ import annotations
from io import StringIO as _StringIO
from types import SimpleNamespace

_embedded_reference = r'''{
  "source": {
    "profile": "onsky",
    "profiles": {
      "internal": {
        "spectral_type": null,
        "temperature_K": 3000,
        "photons_per_second_per_pixel_per_nm": 1000
      },
      "onsky": {
        "spectral_type": "K5V",
        "temperature_K": null,
        "photons_per_second_per_pixel_per_nm": 500
      }
    }
  },
  "spectrum": {
    "mode": "blackbody",
    "weighting": "photon",
    "wavelength_min_m": 1.5e-06,
    "wavelength_max_m": 1.8e-06,
    "samples": 7
  },
  "pupil": {
    "geometry": "solarstein",
    "rotation_deg": 0,
    "array_size": 512,
    "pixels_across_pupil": 128,
    "physical_diameter_m": 1.8
  },
  "flat_dm": {
    "model": "BMC-multi-3.5",
    "flat_command": 0.5,
    "opd_per_command_m": 3e-06,
    "actuator_coupling_factor": 0.75
  },
  "optics": {
    "f_number": 21.2,
    "strehl":1.0
  },
  "phase_mask": {
    "name": "H3",
    "phase_model": "physical_depth",
    "depth_um": 0.654,
    "material": "N_1405",
    "diameter_model": "physical",
    "diameter_um": 44
  },
  "relay": {
    "entrance_diameter_m": 0.012,
    "oap_focal_length_m": 0.254,
    "collimator_focal_length_m": 0.04,
    "distance_to_mirror_m": 0.5,
    "mirror_diameter_m": 0.0254,
    "edge_offset_m": -0.010,
    "edge_angle_rad": 0,
    "imaging_focal_length_m": 0.184,
    "cold_stop_diameter_m": 0.002145,
    "cold_stop_x_offset_m": 0,
    "cold_stop_y_offset_m": 0,
    "detector_pupil_diameter_m": 0.000288,
    "pupil_misconjugation_m": 0
  },
  "detector": {
    "binning": 8,
    "crop": null,
    "exposure_s": 0.001,
    "quantum_efficiency": 0.7
  },
  "pupil_fit": {
        "rotation_coarse_points": 7,
        "geometric_max_function_evaluations": 12,
        "fit_smooth_amplitude": false
    }
}
'''

_embedded_material = r'''Opt. Const. of B-Spline vs. nm
Wavelength(nm)	n	k
245.600189	1.919174	0.116073	
247.187744	1.901554	0.092055	
248.775314	1.884594	0.078349	
250.362961	1.868975	0.074394	
251.950638	1.855674	0.069578	
253.538361	1.843068	0.065176	
255.126129	1.829843	0.063993	
256.713928	1.817092	0.063398	
258.301788	1.805537	0.062747	
259.889709	1.795230	0.063100	
261.477631	1.785339	0.064398	
263.065613	1.775658	0.066319	
264.653656	1.767842	0.068761	
266.241730	1.761750	0.072023	
267.829865	1.755212	0.076724	
269.418030	1.748605	0.082295	
271.006195	1.743731	0.087577	
272.594421	1.740704	0.092767	
274.182709	1.739062	0.098473	
275.771057	1.738871	0.104671	
277.359406	1.740229	0.111045	
278.947815	1.742853	0.116889	
280.536224	1.746329	0.121390	
282.124695	1.750075	0.123570	
283.713196	1.753837	0.122727	
285.301758	1.758384	0.118973	
286.890320	1.764099	0.112689	
288.478943	1.769374	0.104630	
290.067596	1.772504	0.095545	
291.656281	1.773069	0.086037	
293.244995	1.771081	0.076682	
294.833771	1.767027	0.068336	
296.422546	1.761573	0.061915	
298.011353	1.755326	0.057562	
299.600220	1.748851	0.054835	
301.189087	1.742541	0.053309	
302.778015	1.736542	0.052589	
304.366943	1.730996	0.052338	
305.955933	1.726110	0.052380	
307.544922	1.722058	0.052589	
309.133972	1.718587	0.053001	
310.723022	1.715193	0.053748	
312.312103	1.711645	0.054773	
313.901245	1.708197	0.055675	
315.490387	1.705118	0.056108	
317.079559	1.702450	0.056299	
318.668762	1.700132	0.056737	
320.257996	1.698061	0.057663	
321.847229	1.696050	0.058856	
323.436493	1.693928	0.060063	
325.025818	1.691789	0.061220	
326.615143	1.689910	0.062390	
328.204498	1.688475	0.063629	
329.793854	1.687259	0.064942	
331.383240	1.685926	0.066326	
332.972656	1.684343	0.067768	
334.562103	1.682807	0.069242	
336.151550	1.681652	0.070720	
337.741028	1.680994	0.072191	
339.330536	1.680640	0.073670	
340.920044	1.680392	0.075172	
342.509583	1.680192	0.076724	
344.099152	1.680138	0.078359	
345.688721	1.680321	0.080107	
347.278320	1.680739	0.081907	
348.867920	1.681283	0.083604	
350.457550	1.681848	0.085052	
352.047180	1.682414	0.086239	
353.636871	1.683063	0.087316	
355.226532	1.683880	0.088438	
356.816223	1.684908	0.089652	
358.405945	1.686113	0.090811	
359.995636	1.687452	0.091753	
361.585388	1.688887	0.092369	
363.175110	1.690402	0.092736	
364.764862	1.691980	0.092974	
366.354645	1.693612	0.093186	
367.944397	1.695291	0.093354	
369.534210	1.697020	0.093381	
371.124023	1.698798	0.093174	
372.713806	1.700613	0.092685	
374.303650	1.702420	0.091960	
375.893494	1.704171	0.091053	
377.483337	1.705825	0.090012	
379.073181	1.707371	0.088847	
380.663025	1.708817	0.087542	
382.252869	1.710170	0.086080	
383.842743	1.711440	0.084452	
385.432648	1.712631	0.082676	
387.022522	1.713751	0.080774	
388.612396	1.714804	0.078769	
390.202301	1.715782	0.076672	
391.792175	1.716654	0.074483	
393.382080	1.717384	0.072196	
394.971985	1.717939	0.069808	
396.561890	1.718316	0.067338	
398.151794	1.718553	0.064845	
399.741730	1.718689	0.062387	
401.331635	1.718762	0.060020	
402.921570	1.718775	0.057765	
404.511475	1.718673	0.055582	
406.101379	1.718400	0.053426	
407.691284	1.717902	0.051255	
409.281189	1.717174	0.049057	
410.871155	1.716335	0.046886	
412.461029	1.715520	0.044809	
414.050934	1.714856	0.042890	
415.640869	1.714431	0.041176	
417.230774	1.714101	0.039635	
418.820648	1.713642	0.038201	
420.410553	1.712839	0.036812	
422.000427	1.711492	0.035409	
423.590332	1.709618	0.033974	
425.180237	1.707496	0.032543	
426.770081	1.705415	0.031155	
428.359955	1.703647	0.029847	
429.949799	1.702419	0.028652	
431.539642	1.701656	0.027582	
433.129486	1.701166	0.026633	
434.719330	1.700763	0.025805	
436.309174	1.700270	0.025095	
437.898987	1.699563	0.024497	
439.488831	1.698682	0.023984	
441.078613	1.697701	0.023529	
442.668396	1.696689	0.023107	
444.258148	1.695712	0.022692	
445.847961	1.694815	0.022265	
447.437683	1.693984	0.021822	
449.027435	1.693196	0.021362	
450.617126	1.692430	0.020883	
452.206879	1.691665	0.020384	
453.796570	1.690884	0.019868	
455.386292	1.690081	0.019351	
456.975922	1.689252	0.018851	
458.565582	1.688392	0.018387	
460.155212	1.687499	0.017975	
461.744812	1.686571	0.017631	
463.334473	1.685630	0.017341	
464.924011	1.684713	0.017076	
466.513611	1.683851	0.016810	
468.103180	1.683079	0.016518	
469.692719	1.682426	0.016174	
471.282227	1.681897	0.015771	
472.871735	1.681444	0.015335	
474.461182	1.681013	0.014894	
476.050629	1.680552	0.014476	
477.640076	1.680014	0.014107	
479.229492	1.679351	0.013812	
480.818909	1.678573	0.013584	
482.408264	1.677738	0.013394	
483.997620	1.676905	0.013207	
485.586914	1.676129	0.012995	
487.176208	1.675465	0.012727	
488.765472	1.674961	0.012377	
490.354736	1.674604	0.011947	
491.943939	1.674332	0.011461	
493.533142	1.674084	0.010944	
495.122314	1.673801	0.010418	
496.711487	1.673426	0.009905	
498.300598	1.672905	0.009428	
499.889709	1.672231	0.009002	
501.478760	1.671452	0.008637	
503.067780	1.670619	0.008341	
504.656799	1.669782	0.008124	
506.245758	1.668988	0.007994	
507.834717	1.668283	0.007959	
509.423645	1.667703	0.008021	
511.012512	1.667238	0.008151	
512.601379	1.666869	0.008314	
514.190186	1.666579	0.008477	
515.778931	1.666347	0.008606	
517.367737	1.666158	0.008670	
518.956421	1.665994	0.008640	
520.545105	1.665840	0.008503	
522.133789	1.665689	0.008283	
523.722412	1.665533	0.008007	
525.310974	1.665364	0.007701	
526.899536	1.665174	0.007388	
528.488037	1.664957	0.007092	
530.076538	1.664706	0.006836	
531.664978	1.664418	0.006630	
533.253357	1.664100	0.006465	
534.841736	1.663757	0.006328	
536.430054	1.663395	0.006207	
538.018372	1.663020	0.006090	
539.606567	1.662639	0.005967	
541.194824	1.662256	0.005827	
542.782959	1.661875	0.005661	
544.371094	1.661498	0.005476	
545.959167	1.661124	0.005278	
547.547241	1.660753	0.005074	
549.135254	1.660383	0.004871	
550.723206	1.660014	0.004678	
552.311157	1.659647	0.004499	
553.898987	1.659280	0.004342	
555.486816	1.658913	0.004207	
557.074585	1.658540	0.004087	
558.662354	1.658161	0.003976	
560.250061	1.657770	0.003865	
561.837708	1.657365	0.003749	
563.425293	1.656943	0.003620	
565.012817	1.656501	0.003473	
566.600403	1.656036	0.003300	
568.187866	1.655550	0.003105	
569.775269	1.655048	0.002897	
571.362610	1.654537	0.002688	
572.949951	1.654023	0.002491	
574.537231	1.653513	0.002316	
576.124390	1.653011	0.002174	
577.711548	1.652525	0.002075	
579.298706	1.652059	0.002030	
580.885803	1.651617	0.002043	
582.472778	1.651198	0.002102	
584.059753	1.650798	0.002191	
585.646667	1.650413	0.002293	
587.233521	1.650042	0.002392	
588.820313	1.649681	0.002475	
590.407104	1.649328	0.002526	
591.993774	1.648979	0.002531	
593.580383	1.648633	0.002477	
595.166992	1.648287	0.002359	
596.753540	1.647943	0.002189	
598.339966	1.647603	0.001979	
599.926392	1.647267	0.001742	
601.512756	1.646939	0.001490	
603.098999	1.646618	0.001234	
604.685303	1.646306	0.000987	
606.271484	1.646005	0.000759	
607.857544	1.645715	0.000561	
609.443665	1.645437	0.000400	
611.029663	1.645170	0.000274	
612.615601	1.644912	0.000178	
614.201477	1.644660	0.000108	
615.787292	1.644412	0.000059	
617.373047	1.644167	0.000028	
618.958740	1.643923	0.000010	
620.544373	1.643678	0.000002	
622.130005	1.643430	0.000000	
623.715454	1.643179	0.000000	
625.300903	1.642923	0.000000	
626.886292	1.642664	0.000000	
628.471558	1.642405	0.000000	
630.056824	1.642148	0.000000	
631.642029	1.641893	0.000000	
633.227173	1.641643	0.000000	
634.812195	1.641400	0.000000	
636.397156	1.641165	0.000000	
637.982056	1.640939	0.000000	
639.566956	1.640725	0.000000	
641.151733	1.640523	0.000000	
642.736450	1.640332	0.000000	
644.321045	1.640152	0.000000	
645.905640	1.639984	0.000000	
647.490112	1.639825	0.000000	
649.074524	1.639676	0.000000	
650.658875	1.639537	0.000000	
652.243164	1.639407	0.000000	
653.827393	1.639285	0.000000	
655.411499	1.639171	0.000000	
656.995544	1.639065	0.000000	
658.579529	1.638966	0.000000	
660.163452	1.638870	0.000000	
661.747314	1.638776	0.000000	
663.331116	1.638683	0.000000	
664.914734	1.638587	0.000000	
666.498413	1.638487	0.000000	
668.081909	1.638381	0.000000	
669.665405	1.638267	0.000000	
671.248779	1.638143	0.000000	
672.832092	1.638008	0.000000	
674.415283	1.637860	0.000000	
675.998413	1.637699	0.000000	
677.581482	1.637525	0.000000	
679.164490	1.637342	0.000000	
680.747437	1.637151	0.000000	
682.330200	1.636954	0.000000	
683.912964	1.636754	0.000000	
685.495667	1.636551	0.000000	
687.078247	1.636348	0.000000	
688.660767	1.636147	0.000000	
690.243103	1.635948	0.000000	
691.825500	1.635755	0.000000	
693.407715	1.635568	0.000000	
694.989868	1.635388	0.000000	
696.571960	1.635216	0.000000	
698.153931	1.635053	0.000000	
699.735901	1.634899	0.000000	
701.317749	1.634754	0.000000	
702.899475	1.634619	0.000000	
704.481079	1.634494	0.000000	
706.062622	1.634380	0.000000	
707.644165	1.634277	0.000000	
709.225525	1.634185	0.000000	
710.806824	1.634105	0.000000	
712.388062	1.634037	0.000000	
713.969116	1.633981	0.000000	
715.550171	1.633936	0.000000	
717.131104	1.633897	0.000000	
718.711975	1.633860	0.000000	
720.292725	1.633824	0.000000	
721.873413	1.633784	0.000000	
723.453979	1.633737	0.000000	
725.034424	1.633681	0.000000	
726.614807	1.633612	0.000000	
728.195129	1.633527	0.000000	
729.775269	1.633424	0.000000	
731.355408	1.633299	0.000000	
732.935425	1.633150	0.000000	
734.515320	1.632975	0.000000	
736.095154	1.632773	0.000000	
737.674866	1.632550	0.000000	
739.254517	1.632311	0.000000	
740.834045	1.632059	0.000000	
742.413452	1.631801	0.000000	
743.992798	1.631542	0.000000	
745.572021	1.631284	0.000000	
747.151123	1.631034	0.000000	
748.730164	1.630796	0.000000	
750.309082	1.630573	0.000000	
751.887939	1.630370	0.000000	
753.466675	1.630191	0.000000	
755.045288	1.630040	0.000000	
756.623840	1.629921	0.000000	
758.202271	1.629835	0.000000	
759.780579	1.629778	0.000000	
761.358765	1.629743	0.000000	
762.936890	1.629725	0.000000	
764.514893	1.629718	0.000000	
766.092834	1.629716	0.000000	
767.670593	1.629714	0.000000	
769.248291	1.629706	0.000000	
770.825867	1.629688	0.000000	
772.403381	1.629653	0.000000	
773.980774	1.629598	0.000000	
775.557983	1.629518	0.000000	
777.135193	1.629408	0.000000	
778.712280	1.629262	0.000000	
780.289185	1.629078	0.000000	
781.866028	1.628851	0.000000	
783.442749	1.628587	0.000000	
785.019409	1.628293	0.000000	
786.595886	1.627975	0.000000	
788.172241	1.627640	0.000000	
789.748535	1.627293	0.000000	
791.324707	1.626941	0.000000	
792.900757	1.626590	0.000000	
794.476746	1.626246	0.000000	
796.052612	1.625914	0.000000	
797.628296	1.625600	0.000000	
799.203918	1.625310	0.000000	
800.779419	1.625049	0.000000	
802.354797	1.624821	0.000000	
803.930054	1.624633	0.000000	
805.505188	1.624489	0.000000	
807.080200	1.624392	0.000000	
808.655151	1.624340	0.000000	
810.229919	1.624327	0.000000	
811.804565	1.624345	0.000000	
813.379150	1.624390	0.000000	
814.953613	1.624456	0.000000	
816.527954	1.624536	0.000000	
818.102173	1.624626	0.000000	
819.676147	1.624720	0.000000	
821.250183	1.624812	0.000000	
822.824036	1.624899	0.000000	
824.397766	1.624973	0.000000	
825.971375	1.625032	0.000000	
827.544800	1.625069	0.000000	
829.118164	1.625081	0.000000	
830.691406	1.625062	0.000000	
832.264526	1.625009	0.000000	
833.837524	1.624916	0.000000	
835.410339	1.624786	0.000000	
836.983093	1.624621	0.000000	
838.555786	1.624428	0.000000	
840.128235	1.624209	0.000000	
841.700562	1.623970	0.000000	
843.272827	1.623714	0.000000	
844.844910	1.623445	0.000000	
846.416870	1.623168	0.000000	
847.988770	1.622886	0.000000	
849.560425	1.622604	0.000000	
851.132019	1.622324	0.000000	
852.703491	1.622051	0.000000	
854.274902	1.621788	0.000000	
855.846069	1.621538	0.000000	
857.417175	1.621305	0.000000	
858.988098	1.621093	0.000000	
860.558899	1.620904	0.000000	
862.129578	1.620741	0.000000	
863.700073	1.620608	0.000000	
865.270630	1.620502	0.000000	
866.840820	1.620421	0.000000	
868.411011	1.620364	0.000000	
869.981079	1.620328	0.000000	
871.550903	1.620313	0.000000	
873.120605	1.620316	0.000000	
874.690247	1.620336	0.000000	
876.259766	1.620371	0.000000	
877.829102	1.620419	0.000000	
879.398254	1.620479	0.000000	
880.967407	1.620549	0.000000	
882.536255	1.620628	0.000000	
884.105103	1.620714	0.000000	
885.673706	1.620806	0.000000	
887.242188	1.620903	0.000000	
888.810669	1.621002	0.000000	
890.378784	1.621103	0.000000	
891.946960	1.621204	0.000000	
893.514954	1.621303	0.000000	
895.082703	1.621401	0.000000	
896.650391	1.621496	0.000000	
898.217896	1.621587	0.000000	
899.785278	1.621674	0.000000	
901.352539	1.621757	0.000000	
902.919617	1.621834	0.000000	
904.486633	1.621904	0.000000	
906.053406	1.621969	0.000000	
907.620117	1.622026	0.000000	
909.186646	1.622075	0.000000	
910.753052	1.622116	0.000000	
912.319214	1.622149	0.000000	
913.885376	1.622171	0.000000	
915.451294	1.622185	0.000000	
917.017090	1.622187	0.000000	
918.582703	1.622179	0.000000	
920.148132	1.622160	0.000000	
921.713562	1.622128	0.000000	
923.278687	1.622085	0.000000	
924.843750	1.622029	0.000000	
926.408691	1.621959	0.000000	
927.973450	1.621877	0.000000	
929.538086	1.621782	0.000000	
931.102417	1.621677	0.000000	
932.666809	1.621562	0.000000	
934.230957	1.621441	0.000000	
935.794922	1.621313	0.000000	
937.358765	1.621181	0.000000	
938.922485	1.621046	0.000000	
940.485962	1.620909	0.000000	
942.049316	1.620772	0.000000	
943.612610	1.620636	0.000000	
945.175659	1.620503	0.000000	
946.738586	1.620373	0.000000	
948.301392	1.620249	0.000000	
949.863953	1.620131	0.000000	
951.426392	1.620020	0.000000	
952.988647	1.619918	0.000000	
954.550781	1.619825	0.000000	
956.112793	1.619744	0.000000	
957.674622	1.619675	0.000000	
959.236206	1.619619	0.000000	
960.797729	1.619578	0.000000	
962.359070	1.619552	0.000000	
963.920227	1.619542	0.000000	
965.481262	1.619547	0.000000	
967.042114	1.619567	0.000000	
968.602722	1.619599	0.000000	
970.163269	1.619642	0.000000	
971.723633	1.619695	0.000000	
973.283875	1.619757	0.000000	
974.843933	1.619827	0.000000	
976.403809	1.619902	0.000000	
977.963501	1.619983	0.000000	
979.523071	1.620067	0.000000	
981.082397	1.620153	0.000000	
982.641724	1.620241	0.000000	
984.200684	1.620329	0.000000	
985.759583	1.620417	0.000000	
987.318237	1.620502	0.000000	
988.876831	1.620584	0.000000	
990.435242	1.620662	0.000000	
991.993469	1.620735	0.000000	
993.551453	1.620802	0.000000	
995.109375	1.620862	0.000000	
996.667114	1.620914	0.000000	
998.224609	1.620956	0.000000	
999.782043	1.620989	0.000000	
1011.432007	1.620896	0.000000	
1014.815918	1.620777	0.000000	
1018.200195	1.620629	0.000000	
1021.584961	1.620459	0.000000	
1024.970215	1.620272	0.000000	
1028.355957	1.620074	0.000000	
1031.742065	1.619871	0.000000	
1035.128662	1.619668	0.000000	
1038.515869	1.619471	0.000000	
1041.903442	1.619285	0.000000	
1045.291382	1.619114	0.000000	
1048.679932	1.618960	0.000000	
1052.068726	1.618822	0.000000	
1055.458252	1.618698	0.000000	
1058.848022	1.618586	0.000000	
1062.238403	1.618485	0.000000	
1065.629150	1.618392	0.000000	
1069.020264	1.618308	0.000000	
1072.411987	1.618229	0.000000	
1075.804199	1.618155	0.000000	
1079.196777	1.618085	0.000000	
1082.589844	1.618016	0.000000	
1085.983398	1.617948	0.000000	
1089.377441	1.617880	0.000000	
1092.771851	1.617810	0.000000	
1096.166748	1.617738	0.000000	
1099.562134	1.617665	0.000000	
1102.958008	1.617590	0.000000	
1106.354248	1.617514	0.000000	
1109.750977	1.617436	0.000000	
1113.148193	1.617357	0.000000	
1116.545898	1.617277	0.000000	
1119.944092	1.617196	0.000000	
1123.342529	1.617113	0.000000	
1126.741699	1.617030	0.000000	
1130.141235	1.616946	0.000000	
1133.541138	1.616861	0.000000	
1136.941650	1.616775	0.000000	
1140.342529	1.616689	0.000000	
1143.743896	1.616602	0.000000	
1147.145630	1.616514	0.000000	
1150.547852	1.616426	0.000000	
1153.950562	1.616338	0.000000	
1157.353760	1.616249	0.000000	
1160.757446	1.616159	0.000000	
1164.161499	1.616070	0.000000	
1167.566162	1.615980	0.000000	
1170.971191	1.615890	0.000000	
1174.376709	1.615799	0.000000	
1177.782593	1.615709	0.000000	
1181.188965	1.615618	0.000000	
1184.595947	1.615527	0.000000	
1188.003174	1.615436	0.000000	
1191.410889	1.615345	0.000000	
1194.819214	1.615254	0.000000	
1198.227905	1.615163	0.000000	
1201.637085	1.615072	0.000000	
1205.046631	1.614982	0.000000	
1208.456787	1.614892	0.000000	
1211.867310	1.614802	0.000000	
1215.278320	1.614714	0.000000	
1218.689697	1.614626	0.000000	
1222.101807	1.614539	0.000000	
1225.514038	1.614454	0.000000	
1228.927002	1.614369	0.000000	
1232.340210	1.614287	0.000000	
1235.754028	1.614205	0.000000	
1239.168213	1.614125	0.000000	
1242.582886	1.614047	0.000000	
1245.998047	1.613971	0.000000	
1249.413574	1.613897	0.000000	
1252.829712	1.613825	0.000000	
1256.246216	1.613755	0.000000	
1259.663208	1.613687	0.000000	
1263.080566	1.613621	0.000000	
1266.498413	1.613557	0.000000	
1269.916748	1.613496	0.000000	
1273.335693	1.613436	0.000000	
1276.754883	1.613378	0.000000	
1280.174561	1.613321	0.000000	
1283.594849	1.613266	0.000000	
1287.015381	1.613213	0.000000	
1290.436523	1.613161	0.000000	
1293.858154	1.613110	0.000000	
1297.280151	1.613061	0.000000	
1300.702637	1.613013	0.000000	
1304.125610	1.612966	0.000000	
1307.548950	1.612921	0.000000	
1310.972778	1.612876	0.000000	
1314.397095	1.612832	0.000000	
1317.821899	1.612790	0.000000	
1321.247192	1.612748	0.000000	
1324.672852	1.612706	0.000000	
1328.098877	1.612666	0.000000	
1331.525635	1.612626	0.000000	
1334.952637	1.612586	0.000000	
1338.380127	1.612547	0.000000	
1341.808228	1.612507	0.000000	
1345.236572	1.612467	0.000000	
1348.665527	1.612426	0.000000	
1352.094849	1.612385	0.000000	
1355.524658	1.612343	0.000000	
1358.954956	1.612300	0.000000	
1362.385742	1.612256	0.000000	
1365.816895	1.612210	0.000000	
1369.248535	1.612163	0.000000	
1372.680664	1.612114	0.000000	
1376.113281	1.612064	0.000000	
1379.546265	1.612012	0.000000	
1382.979736	1.611958	0.000000	
1386.413696	1.611901	0.000000	
1389.848145	1.611843	0.000000	
1393.282959	1.611782	0.000000	
1396.718384	1.611718	0.000000	
1400.154053	1.611652	0.000000	
1403.590332	1.611584	0.000000	
1407.027100	1.611513	0.000000	
1410.464111	1.611440	0.000000	
1413.901855	1.611364	0.000000	
1417.339844	1.611287	0.000000	
1420.778442	1.611209	0.000000	
1424.217407	1.611128	0.000000	
1427.656860	1.611046	0.000000	
1431.096802	1.610963	0.000000	
1434.537109	1.610879	0.000000	
1437.977905	1.610793	0.000000	
1441.419189	1.610707	0.000000	
1444.860962	1.610620	0.000000	
1448.303101	1.610532	0.000000	
1451.745850	1.610444	0.000000	
1455.188965	1.610355	0.000000	
1458.632568	1.610266	0.000000	
1462.076538	1.610177	0.000000	
1465.520996	1.610088	0.000000	
1468.966064	1.609998	0.000000	
1472.411377	1.609909	0.000000	
1475.857178	1.609820	0.000000	
1479.303589	1.609731	0.000000	
1482.750366	1.609643	0.000000	
1486.197510	1.609555	0.000000	
1489.645264	1.609468	0.000000	
1493.093506	1.609382	0.000000	
1496.542114	1.609297	0.000000	
1499.991089	1.609212	0.000000	
1503.440674	1.609129	0.000000	
1506.890625	1.609048	0.000000	
1510.341064	1.608968	0.000000	
1513.791992	1.608889	0.000000	
1517.243408	1.608812	0.000000	
1520.695068	1.608737	0.000000	
1524.147461	1.608665	0.000000	
1527.600220	1.608594	0.000000	
1531.053345	1.608525	0.000000	
1534.507080	1.608459	0.000000	
1537.961182	1.608395	0.000000	
1541.415649	1.608334	0.000000	
1544.870728	1.608275	0.000000	
1548.326172	1.608219	0.000000	
1551.782104	1.608166	0.000000	
1555.238525	1.608116	0.000000	
1558.695313	1.608068	0.000000	
1562.152588	1.608024	0.000000	
1565.610474	1.607983	0.000000	
1569.068604	1.607945	0.000000	
1572.527344	1.607910	0.000000	
1575.986450	1.607879	0.000000	
1579.446045	1.607851	0.000000	
1582.906128	1.607827	0.000000	
1586.366577	1.607805	0.000000	
1589.827637	1.607787	0.000000	
1593.289063	1.607771	0.000000	
1596.750854	1.607758	0.000000	
1600.213257	1.607748	0.000000	
1603.676025	1.607740	0.000000	
1607.139282	1.607735	0.000000	
1610.603027	1.607731	0.000000	
1614.067139	1.607730	0.000000	
1617.531860	1.607731	0.000000	
1620.996826	1.607733	0.000000	
1624.462402	1.607737	0.000000	
1627.928467	1.607742	0.000000	
1631.394897	1.607749	0.000000	
1634.861816	1.607758	0.000000	
1638.329224	1.607767	0.000000	
1641.797119	1.607778	0.000000	
1645.265259	1.607790	0.000000	
1648.734009	1.607802	0.000000	
1652.203369	1.607815	0.000000	
1655.672852	1.607829	0.000000	
1659.142944	1.607844	0.000000	
1662.613647	1.607859	0.000000	
1666.084595	1.607875	0.000000	
1669.556030	1.607891	0.000000	
1673.028076	1.607907	0.000000	
1676.500366	1.607924	0.000000	
1679.973267	1.607940	0.000000	
1683.446655	1.607957	0.000000	
			'''

ASGARD_ALIGNMENT_LICENSE = r'''Copyright (c) 2018 The Python Packaging Authority

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.'''


# Pupil geometry

import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares

def estimate_pupil_center_subpixel(image: np.ndarray) -> dict:
    """Fit the outer pupil edge with a robust circle at subpixel precision.

    The fit uses only strong gradients in an annulus around the outer pupil
    boundary.  This suppresses the central obstruction, spiders, and smooth
    illumination structure, none of which should define the pupil centre.
    """
    image = np.asarray(image, dtype=float)
    if image.ndim != 2 or not np.all(np.isfinite(image)):
        raise ValueError('The pupil-centre image must be a finite 2-D array.')
    border = np.concatenate((image[0], image[-1], image[:, 0], image[:, -1]))
    background = float(np.median(border))
    mad = float(np.median(np.abs(border - background)))
    noise_sigma = max(1.4826 * mad, np.finfo(float).eps)
    signal = np.clip(image - background, 0.0, None)
    smooth = ndimage.gaussian_filter(signal, 0.55)
    peak = float(smooth.max())
    if not np.isfinite(peak) or peak <= 0.0:
        raise RuntimeError('The clear pupil contains no positive signal.')
    threshold = max(5.0 * noise_sigma, 0.08 * peak)
    labels, count = ndimage.label(smooth > threshold)
    if count == 0:
        raise RuntimeError('Could not identify the clear-pupil support.')
    sizes = np.asarray(ndimage.sum(np.ones_like(labels), labels, range(1, count + 1)), dtype=float)
    largest = float(sizes.max())
    retained = np.flatnonzero(sizes >= max(4.0, 0.08 * largest)) + 1
    support = np.isin(labels, retained)
    rows, columns = np.nonzero(support)
    if rows.size < 12:
        raise RuntimeError('Too few illuminated pixels for a pupil-centre fit.')
    center_y0 = 0.5 * (rows.min() + rows.max())
    center_x0 = 0.5 * (columns.min() + columns.max())
    radius0 = 0.25 * (rows.max() - rows.min() + columns.max() - columns.min() + 2.0)
    yy, xx = np.indices(image.shape, dtype=float)
    radius = np.hypot(xx - center_x0, yy - center_y0)
    gradient_y = ndimage.sobel(smooth, axis=0, mode='nearest') / 8.0
    gradient_x = ndimage.sobel(smooth, axis=1, mode='nearest') / 8.0
    gradient = np.hypot(gradient_x, gradient_y)
    annulus_half_width = max(1.5, 0.18 * radius0)
    annulus = np.abs(radius - radius0) <= annulus_half_width
    candidate_gradient = gradient[annulus]
    positive = candidate_gradient[candidate_gradient > 0.0]
    if positive.size < 12:
        raise RuntimeError('The outer pupil edge is too weak for a centre fit.')
    gradient_threshold = max(float(np.percentile(positive, 55.0)), 3.0 * noise_sigma)
    edge = annulus & (gradient >= gradient_threshold)
    edge_y, edge_x = np.nonzero(edge)
    if edge_y.size < 12:
        raise RuntimeError('Too few outer-edge samples for a pupil-centre fit.')
    weights = gradient[edge]
    weights /= max(float(np.median(weights)), np.finfo(float).eps)
    sqrt_weights = np.sqrt(np.clip(weights, 0.1, 10.0))

    def residual(parameters):
        center_x, center_y, fitted_radius = parameters
        distances = np.hypot(edge_x - center_x, edge_y - center_y)
        return sqrt_weights * (distances - fitted_radius)
    maximum_radius = float(np.hypot(*image.shape))
    result = least_squares(residual, np.array([center_x0, center_y0, radius0]), bounds=(np.array([0.0, 0.0, 1.0]), np.array([image.shape[1] - 1.0, image.shape[0] - 1.0, maximum_radius])), loss='soft_l1', f_scale=0.35, x_scale=np.array([0.25, 0.25, 0.5]), max_nfev=200, ftol=1e-11, xtol=1e-11, gtol=1e-11)
    center_x, center_y, fitted_radius = map(float, result.x)
    profile_radius = np.hypot(xx - center_x, yy - center_y)
    profile_region = np.abs(profile_radius - fitted_radius) <= max(3.0, 0.4 * fitted_radius)
    profile_x = xx[profile_region]
    profile_y = yy[profile_region]
    profile_data = smooth[profile_region]
    residual_scale = max(noise_sigma, 0.001 * peak)

    def profile_residual(parameters):
        trial_x, trial_y, trial_radius, edge_width, level, slope_x, slope_y, local_background = parameters
        distance = np.hypot(profile_x - trial_x, profile_y - trial_y)
        transition = 0.5 * (1.0 - np.tanh((distance - trial_radius) / edge_width))
        illumination = level + slope_x * (profile_x - trial_x) / trial_radius + slope_y * (profile_y - trial_y) / trial_radius
        return (local_background + illumination * transition - profile_data) / residual_scale
    profile_result = least_squares(profile_residual, np.array([center_x, center_y, fitted_radius, 0.6, peak, 0.0, 0.0, 0.0]), bounds=(np.array([center_x - 2.0, center_y - 2.0, 0.7 * fitted_radius, 0.15, 0.2 * peak, -peak, -peak, -0.2 * peak]), np.array([center_x + 2.0, center_y + 2.0, 1.3 * fitted_radius, 2.0, 2.0 * peak, peak, peak, 0.2 * peak])), loss='soft_l1', f_scale=2.0, x_scale=np.array([0.2, 0.2, 0.5, 0.2, peak, 0.2 * peak, 0.2 * peak, residual_scale]), max_nfev=500, ftol=1e-11, xtol=1e-11, gtol=1e-11)
    center_x, center_y, fitted_radius = map(float, profile_result.x[:3])
    angles = np.mod(np.arctan2(edge_y - center_y, edge_x - center_x), 2 * np.pi)
    occupied = np.unique(np.floor(angles / (2 * np.pi) * 24).astype(int))
    angular_coverage = float(occupied.size / 24.0)
    if not profile_result.success:
        raise RuntimeError(f'Subpixel pupil-centre fit failed: {profile_result.message}')
    if angular_coverage < 0.35:
        raise RuntimeError(f'Subpixel pupil-centre fit has insufficient outer-edge coverage ({angular_coverage:.0%}).')
    radial_residual = np.hypot(edge_x - center_x, edge_y - center_y) - fitted_radius
    radial_rms = float(np.sqrt(np.mean(radial_residual ** 2)))
    center_sigma_x = np.nan
    center_sigma_y = np.nan
    if profile_result.jac.shape[0] > profile_result.jac.shape[1]:
        try:
            covariance = np.linalg.inv(profile_result.jac.T @ profile_result.jac)
            residual_variance = float(np.sum(profile_result.fun ** 2) / (profile_result.jac.shape[0] - profile_result.jac.shape[1]))
            covariance *= residual_variance
            center_sigma_x = float(np.sqrt(max(covariance[0, 0], 0.0)))
            center_sigma_y = float(np.sqrt(max(covariance[1, 1], 0.0)))
        except np.linalg.LinAlgError:
            pass
    return {'center_x': center_x, 'center_y': center_y, 'radius_pixels': fitted_radius, 'center_sigma_x_pixels': center_sigma_x, 'center_sigma_y_pixels': center_sigma_y, 'radial_residual_rms_pixels': radial_rms, 'angular_coverage_fraction': angular_coverage, 'edge_sample_count': int(edge_y.size), 'method': 'robust outer-edge profile with planar illumination', 'fit_success': bool(profile_result.success), 'fit_message': str(profile_result.message)}


# Halo estimator

import numpy as np


def measure_halo(cube_e, clear_e, pupil, read_noise_e, clear_frames,
                 background_e=0., calibration=None, weights=None):
    cube=np.asarray(cube_e,float)
    if cube.ndim==2:
        cube=cube[None]
    clear=np.asarray(clear_e,float)
    pupil=np.asarray(pupil,bool)
    if cube.ndim!=3 or cube.shape[1:]!=clear.shape or pupil.shape!=clear.shape:
        raise ValueError('Halo image and pupil shapes differ')
    if not pupil.any() or pupil.all():
        raise ValueError('Halo diagnostic needs inside and outside pixels')
    weights=np.ones_like(clear) if weights is None else np.asarray(weights,float)
    if weights.shape!=clear.shape or not np.all(np.isfinite(weights)) or np.any((weights<0)|(weights>1)):
        raise ValueError('Pixel weights must be finite, matching shape, in [0,1]')
    coefficients=np.zeros_like(clear)
    for region in (pupil,~pupil):
        if np.sum(weights[region])<max(1.,.5*region.sum()):
            raise ValueError('Too little reliable pixel coverage in a pupil region (<50%)')
        # Preserve area for a uniform field; missing structure remains a systematic.
        coefficients[region]=weights[region]*region.sum()/np.sum(weights[region])
    used=coefficients>0
    valid=np.all(np.isfinite(cube[:,used]),axis=1)
    cube=np.where(used,cube,0.)
    clear=np.where(used,clear,0.)
    inside=np.sum(cube[:,pupil]*coefficients[pupil],axis=1)
    outside=np.sum(cube[:,~pupil]*coefficients[~pupil],axis=1)
    total=inside+outside
    noise=(np.maximum(cube,0)+read_noise_e**2+background_e)*coefficients**2
    vi=noise[:,pupil].sum(axis=1)
    vo=noise[:,~pupil].sum(axis=1)
    with np.errstate(divide='ignore',invalid='ignore'):
        ratio=outside/inside
        fraction=outside/total
        sigma=np.sqrt(inside**2*vo+outside**2*vi)/total**2
        ratio_sigma=np.sqrt(vo/inside**2+outside**2*vi/inside**4)
    valid &= (total>5*np.sqrt(vi+vo)) & (inside>5*np.sqrt(vi))
    ci,co=float((clear*coefficients)[pupil].sum()),float((clear*coefficients)[~pupil].sum())
    ct=ci+co
    if not np.isfinite(ct) or ct<=0:
        raise ValueError('Clear reference has no positive finite total flux')
    baseline=co/ct
    cn=(np.maximum(clear,0)+read_noise_e**2+background_e)*coefficients**2/max(clear_frames,1)
    baseline_sigma=float(np.sqrt(ci**2*cn[~pupil].sum()+co**2*cn[pupil].sum())/ct**2)
    excess=fraction-baseline
    excess_sigma=np.sqrt(sigma**2+baseline_sigma**2) if clear_frames>0 else np.full_like(sigma,np.nan)
    strehl=np.full_like(fraction,np.nan)
    strehl_sigma=strehl.copy()
    if calibration is not None:
        delta=calibration['fraction']-baseline
        if delta<=0:
            raise ValueError('Halo reference must exceed the clear outside fraction')
        sr=calibration['strehl']
        strehl=sr*excess/delta
        # Same clear baseline appears in numerator and denominator: keep covariance.
        strehl_sigma=abs(sr)*np.sqrt((sigma/delta)**2+
            (excess*calibration['fraction_sigma']/delta**2)**2+
            ((fraction-calibration['fraction'])*baseline_sigma/delta**2)**2)
        if clear_frames<=0:
            strehl_sigma[:]=np.nan
    for values in (ratio,fraction,sigma,ratio_sigma,excess,excess_sigma,strehl,strehl_sigma):
        values[~valid]=np.nan
    return dict(inside_e=inside,outside_e=outside,ratio=ratio,ratio_sigma=ratio_sigma,
        fraction=fraction,fraction_sigma=sigma,excess_fraction=excess,
        excess_sigma=excess_sigma,strehl=strehl,strehl_sigma=strehl_sigma,
        valid=valid,clear_fraction=baseline,clear_fraction_sigma=baseline_sigma,
        calibration=calibration,pixel_weights=weights,inside_pixels=int(pupil.sum()),outside_pixels=int((~pupil).sum()))


def reference_diagnostics(measurement):
    """Explain the calibration denominator and its conditional statistical error."""
    valid=measurement['valid']
    values=measurement['fraction'][valid]
    sigmas=measurement['fraction_sigma'][valid]
    if len(values)<3:
        raise ValueError(f'Only {len(values)}/{len(valid)} valid frames; need at least three. '
                         'Check total/pupil signal, bias, gain and read noise.')
    mean=float(values.mean())
    error=float(max(np.sqrt(np.sum(sigmas**2))/len(values),values.std(ddof=1)/np.sqrt(len(values))))
    baseline=float(measurement['clear_fraction'])
    baseline_error=float(measurement['clear_fraction_sigma'])
    delta=mean-baseline
    uncertainty=float(np.hypot(error,baseline_error))
    if not np.isfinite(uncertainty) or not np.isfinite(delta):
        raise ValueError('Nonfinite reference uncertainty; check clear reference and noise settings')
    significance=delta/uncertainty if uncertainty>0 else (float('inf') if delta>0 else 0.)
    return dict(fraction=mean,fraction_sigma=error,clear_fraction=baseline,
                clear_fraction_sigma=baseline_error,excess=delta,
                excess_sigma=uncertainty,significance=significance,frames=len(values))


def reference_from_window(measurement,strehl):
    """Reference uncertainty includes observed frame scatter; assumes independent samples."""
    if not np.isfinite(strehl) or not 0<strehl<=1:
        raise ValueError('Reference Strehl must be in (0,1]')
    d=reference_diagnostics(measurement)
    if d['excess']<=0 or d['significance']<=5:
        detail=(f"outside fraction IN={d['fraction']:.5f}, clear={d['clear_fraction']:.5f}; "
                f"excess={d['excess']:.5f} ± {d['excess_sigma']:.5f} "
                f"({d['significance']:.2f} sigma; need >5). "
                f"Uncertainty: IN={d['fraction_sigma']:.5f}, clear={d['clear_fraction_sigma']:.5f}. ")
        if d['excess']<=0:
            advice=('No positive mask-induced halo excess. Check mask OUT/IN states, '
                    'pupil circle, crop and matching source/exposure.')
        else:
            advice=('Positive excess is too uncertain. Verify detector noise/bias and matching '
                    'references; use a longer --frames window and recapture BOTH references.')
        raise ValueError(detail+advice)
    return dict(fraction=d['fraction'],fraction_sigma=d['fraction_sigma'],
                strehl=float(strehl),frames=d['frames'],significance=d['significance'])


# Detector

import numpy as np
from scipy import ndimage


def fill_for_geometry(image, good):
    """Nearest-neighbor fill is for fitting only; excluded pixels stay excluded."""
    if not good.any():
        raise ValueError('No usable clear pixels')
    nearest = ndimage.distance_transform_edt(~good, return_distances=False, return_indices=True)
    return image[tuple(nearest)]


def estimate_detector(clear_adu, gain=2., buffer_px=3., bad_pixels=None):
    """Return one scalar background, frozen pixel mask and effective noise.

    Background is the median over all finite exterior samples in the clear cube.
    Noise is MAD of non-overlapping frame-pair differences / sqrt(2). Static
    offsets cancel in these differences. This is background+read noise, not a
    separately identifiable electronic read-noise measurement.
    """
    cube = np.asarray(clear_adu, float)
    if cube.ndim != 3 or len(cube) < 6:
        raise ValueError('Use at least six clear frames to estimate background/noise')
    if not np.isfinite(gain) or gain <= 0 or not np.isfinite(buffer_px) or buffer_px <= 0:
        raise ValueError('Gain and exterior buffer must be finite and positive')
    good = np.isfinite(cube).all(axis=0)
    if bad_pixels is not None:
        if np.shape(bad_pixels) != cube.shape[1:]:
            raise ValueError('Bad-pixel mask shape mismatch')
        good &= ~np.asarray(bad_pixels, bool)
    mean = np.where(good, cube, 0.).mean(axis=0)
    geometry = estimate_pupil_center_subpixel(fill_for_geometry(mean, good))
    y, x = np.indices(mean.shape)
    radius = np.hypot(x-geometry['center_x'], y-geometry['center_y'])
    outside = good & (radius > geometry['radius_pixels']+buffer_px)
    if outside.sum() < 20:
        raise ValueError('Fewer than 20 background pixels outside pupil + buffer')
    background = float(np.median(cube[:, outside]))  # Exactly one estimate per run.
    pairs = len(cube)//2
    differences = (cube[1:2*pairs:2, outside]-cube[:2*pairs:2, outside])/np.sqrt(2.)
    center = np.median(differences)
    sigma_adu = float(1.4826022185*np.median(abs(differences-center)))
    if not np.isfinite(sigma_adu):
        raise ValueError('Nonfinite estimated noise')
    # Means/medians are saved for diagnosing drift, NEVER used to resubtract frames.
    frame_medians = np.median(cube[:, outside], axis=1)
    return dict(background_adu=background, effective_noise_adu=sigma_adu,
                effective_noise_e=sigma_adu*gain, background_pixels=int(outside.sum()),
                background_buffer_px=buffer_px, clear_frames=len(cube),
                background_frame_scatter_adu=float(np.std(frame_medians, ddof=1)),
                geometry=geometry), good, outside


def reduce_frames(raw, background_adu, gain, good):
    raw = np.asarray(raw, float)
    if raw.ndim != 3 or raw.shape[1:] != good.shape or not np.isfinite(raw[:, good]).all():
        raise ValueError('Camera shape changed or good pixels became nonfinite')
    return np.where(good, (raw-background_adu)*gain, 0.)


# Registration

import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import RectBivariateSpline

def get_inner_square_indices(outer_size, inner_offset, without_outer_corners=True):
    if without_outer_corners:
        grid = np.arange(outer_size ** 2).reshape(outer_size, outer_size).astype(float)
        grid[0, 0] = grid[0, -1] = grid[-1, 0] = grid[-1, -1] = np.nan
        top_row = inner_offset
        bottom_row = outer_size - inner_offset - 1
        left_col = inner_offset
        right_col = outer_size - inner_offset - 1
        inner_corners = [grid[top_row, left_col], grid[top_row, right_col], grid[bottom_row, left_col], grid[bottom_row, right_col]]
        valid_indices = np.isfinite(grid.flatten())
        valid_actuators = np.where(valid_indices)[0]
        inner_square_indices = [np.where(valid_actuators == np.where(grid.flatten() == corner)[0][0])[0][0] for corner in inner_corners]
    else:
        grid = np.arange(outer_size ** 2).reshape(outer_size, outer_size)
        top_row = inner_offset
        bottom_row = outer_size - inner_offset - 1
        left_col = inner_offset
        right_col = outer_size - inner_offset - 1
        inner_corners = [grid[top_row, left_col], grid[top_row, right_col], grid[bottom_row, left_col], grid[bottom_row, right_col]]
        inner_square_indices = [np.where(grid.flatten() == corner)[0][0] for corner in inner_corners]
    return inner_square_indices

def gaussian_2d(xy, amp, x0, y0, sigma_x, sigma_y, theta, offset):
    x, y = xy
    x0, y0 = (float(x0), float(y0))
    a = np.cos(theta) ** 2 / (2 * sigma_x ** 2) + np.sin(theta) ** 2 / (2 * sigma_y ** 2)
    b = -np.sin(2 * theta) / (4 * sigma_x ** 2) + np.sin(2 * theta) / (4 * sigma_y ** 2)
    c = np.sin(theta) ** 2 / (2 * sigma_x ** 2) + np.cos(theta) ** 2 / (2 * sigma_y ** 2)
    return offset + amp * np.exp(-(a * (x - x0) ** 2 + 2 * b * (x - x0) * (y - y0) + c * (y - y0) ** 2))

def get_adjacent_pixels(coord, radius, pixel_values):
    """
    I, J: Coordinates of the central pixel
    radius: Radius (in pixels) for the region of interest
    pixel_values: 2D array of the image pixel values
    
    Returns:
    - Adjacent pixel indices and values within the radius
    """
    n, m = coord
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    mask = x ** 2 + y ** 2 <= radius ** 2
    indices = np.argwhere(mask)
    indices[:, 0] += n - radius
    indices[:, 1] += m - radius
    valid_indices = [(i, j) for i, j in indices if i >= 0 and j >= 0 and (i < pixel_values.shape[0]) and (j < pixel_values.shape[1])]
    pixel_values_adj = np.array([pixel_values[i, j] for i, j in valid_indices])
    return (np.array(valid_indices), pixel_values_adj)

def interpolate_and_fit_gaussian(coord, radius, pixel_values, factor=5):
    """
    coord = tuple of coordinates of the central pixel (row, column)
    radius: Radius of the neighborhood
    pixel_values: 2D array of pixel values (image)
    factor: Upsampling factor for spline interpolation (default 5x finer grid)
    
    Returns:
    A dictionary with fitted Gaussian parameters and residuals.
    """
    i, j = coord
    indices, pixel_values_adj = get_adjacent_pixels((i, j), radius, pixel_values)
    x = indices[:, 1]
    y = indices[:, 0]
    x_grid = np.unique(x)
    y_grid = np.unique(y)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = pixel_values[np.ix_(y_grid, x_grid)]
    spline = RectBivariateSpline(y_grid, x_grid, Z)
    x_fine = np.linspace(x_grid.min(), x_grid.max(), factor * len(x_grid))
    y_fine = np.linspace(y_grid.min(), y_grid.max(), factor * len(y_grid))
    X_fine, Y_fine = np.meshgrid(x_fine, y_fine)
    Z_fine = spline(y_fine, x_fine)
    initial_guess = (Z_fine.max(), np.mean(x_fine), np.mean(y_fine), 1, 1, 0, np.median(Z_fine))
    try:
        popt, pcov = curve_fit(gaussian_2d, (X_fine.ravel(), Y_fine.ravel()), Z_fine.ravel(), p0=initial_guess)
    except RuntimeError:
        return {'error': 'Fit did not converge'}
    Z_fit = gaussian_2d((X_fine, Y_fine), *popt).reshape(X_fine.shape)
    residuals = Z_fine - Z_fit
    amp_fit, x0_fit, y0_fit, sigma_x_fit, sigma_y_fit, theta_fit, offset_fit = popt
    x0_global = x0_fit
    y0_global = y0_fit
    fit_dict = {'amplitude': amp_fit, 'x0': x0_global, 'y0': y0_global, 'sigma_x': sigma_x_fit, 'sigma_y': sigma_y_fit, 'theta': theta_fit, 'offset': offset_fit, 'X': X, 'Y': Y, 'I_raw': Z, 'X_interp': X_fine, 'Y_interp': Y_fine, 'I_interp': Z_fine, 'I_fit': Z_fit, 'residuals': residuals}
    return fit_dict

def find_intersection(x1, y1, x2, y2, x3, y3, x4, y4):
    """
    Find the intersection point of two lines:
    Line 1: passing through (x1, y1) and (x2, y2)
    Line 2: passing through (x3, y3) and (x4, y4)
    """
    a1 = y2 - y1
    b1 = x1 - x2
    c1 = a1 * x1 + b1 * y1
    a2 = y4 - y3
    b2 = x3 - x4
    c2 = a2 * x3 + b2 * y3
    det = a1 * b2 - a2 * b1
    if det == 0:
        raise ValueError('The lines are parallel and do not intersect.')
    x_intersect = (b2 * c1 - b1 * c2) / det
    y_intersect = (a1 * c2 - a2 * c1) / det
    return (x_intersect, y_intersect)

def sort_corners(corners):
    corners = np.array(corners)
    centroid = np.mean(corners, axis=0)

    def angle_from_centroid(point):
        return np.arctan2(point[1] - centroid[1], point[0] - centroid[0])
    sorted_corners = sorted(corners, key=angle_from_centroid)
    top_left = sorted_corners[3]
    top_right = sorted_corners[2]
    bottom_right = sorted_corners[1]
    bottom_left = sorted_corners[0]
    return (top_left, top_right, bottom_left, bottom_right)

def find_quadrilateral_diagonal_intersection(corners):
    """
    Given four corners of a quadrilateral, find the intersection point of its diagonals.
    
    Args:
    - corners: A list of 4 tuples, each representing the (x, y) coordinates of a corner.
    
    Output:
    - Intersection point (x, y) of the diagonals
    """
    top_left, top_right, bottom_left, bottom_right = sort_corners(corners)
    intersection = find_intersection(top_left[0], top_left[1], bottom_right[0], bottom_right[1], top_right[0], top_right[1], bottom_left[0], bottom_left[1])
    return intersection

def generate_dm_coordinates(Nx=12, Ny=12, spacing=1.0):
    """
    Generates the x, y coordinates of the actuators in a 12x12 grid DM with missing corners.
    
    Args:
        Nx, Ny: Number of actuators in the x and y directions (12x12 grid).
        spacing: The spacing between actuators (default is 1 unit).
    
    Returns:
        - coords: A list of tuples (x, y) representing the coordinates of the actuators.
        - flattened_indices: A dictionary that maps actuator indices (0 to 139) to (x, y) coordinates.
        - coord_to_index: A dictionary mapping (x, y) coordinates to actuator indices.
    """
    coords = []
    coord_to_index = {}
    flattened_indices = {}
    center_x = (Nx - 1) / 2
    center_y = (Ny - 1) / 2
    actuator_index = 0
    for i in range(Ny):
        for j in range(Nx):
            if i == 0 and j == 0 or (i == 0 and j == Nx - 1) or (i == Ny - 1 and j == 0) or (i == Ny - 1 and j == Nx - 1):
                continue
            x = (j - center_x) * spacing
            y = (i - center_y) * spacing
            coords.append((x, y))
            coord_to_index[x, y] = actuator_index
            flattened_indices[actuator_index] = (x, y)
            actuator_index += 1
    return (coords, flattened_indices, coord_to_index)

def fit_affine_transformation_with_center(corners_dm, corners_img, intersection_img):
    """
    Fit an affine transformation from DM space to image space, using the DM center as the origin (0,0).
    
    Args:
        corners_dm: List of (x, y) coordinates of DM corners in DM space (relative to the DM center).
        corners_img: List of (x, y) coordinates of the corresponding points in image space.
        intersection_img: The (x, y) coordinates of the DM center in image space.
    
    Returns:
        - transform_matrix: A 2x3 matrix that transforms DM coordinates to pixel coordinates.
    """
    dm = np.array(corners_dm)
    img = np.array(corners_img)
    tx, ty = intersection_img
    dm_coords = np.vstack([dm.T, np.ones(len(dm))]).T
    img_coords = img - np.array([tx, ty])
    A_x = np.linalg.lstsq(dm_coords[:, :2], img_coords[:, 0], rcond=None)[0]
    A_y = np.linalg.lstsq(dm_coords[:, :2], img_coords[:, 1], rcond=None)[0]
    transform_matrix = np.array([[A_x[0], A_x[1], tx], [A_y[0], A_y[1], ty]])
    return transform_matrix

def dm_to_pixel(dm_coord, transform_matrix):
    """
    Converts DM coordinates to pixel coordinates using the affine transformation.
    
    Args:
        dm_coord: A tuple (x, y) in DM space.
        transform_matrix: The affine transformation matrix from DM space to pixel space.
    
    Returns:
        Tuple (x_pixel, y_pixel) in pixel coordinates.
    """
    dm_coord = np.array(dm_coord)
    pixel_coord = np.dot(transform_matrix[:, :2], dm_coord) + transform_matrix[:, 2]
    return tuple(pixel_coord)


def calibrate_transform_between_DM_and_image(indices, images, debug=False):
    """Preserve the pipeline Gaussian/affine registration, without plots or files."""
    fitted = []
    for image in images:
        peak = np.unravel_index(np.argmax(abs(image)), image.shape)
        fit = interpolate_and_fit_gaussian(peak, 5, abs(image), factor=5)
        if 'error' in fit:
            raise RuntimeError(fit['error'])
        fitted.append([fit['x0'], fit['y0']])
    observed = np.asarray(fitted)
    center = find_quadrilateral_diagonal_intersection(observed)
    coords = np.asarray(generate_dm_coordinates()[0])
    matrix = fit_affine_transformation_with_center(coords[indices], observed, center)
    return dict(img_corners=observed,
                actuator_coord_list_pixel_space=np.array([dm_to_pixel(c, matrix) for c in coords]),
                actuator_to_pixel_matrix=matrix)


# Retain the registration interface used by register_dm without a module import.
registration = SimpleNamespace(
    get_inner_square_indices=get_inner_square_indices,
    gaussian_2d=gaussian_2d,
    get_adjacent_pixels=get_adjacent_pixels,
    interpolate_and_fit_gaussian=interpolate_and_fit_gaussian,
    find_intersection=find_intersection,
    sort_corners=sort_corners,
    find_quadrilateral_diagonal_intersection=find_quadrilateral_diagonal_intersection,
    generate_dm_coordinates=generate_dm_coordinates,
    fit_affine_transformation_with_center=fit_affine_transformation_with_center,
    dm_to_pixel=dm_to_pixel,
    calibrate_transform_between_DM_and_image=calibrate_transform_between_DM_and_image,
)

# Reference model

import json
from pathlib import Path
import numpy as np
from scipy import ndimage
from scipy.interpolate import interp1d
PHASE_MASK_PRESETS = {'J1': {'depth_um': 0.474, 'diameter_um': 32.0}, 'J2': {'depth_um': 0.474, 'diameter_um': 36.0}, 'J3': {'depth_um': 0.474, 'diameter_um': 44.0}, 'J4': {'depth_um': 0.474, 'diameter_um': 54.0}, 'J5': {'depth_um': 0.474, 'diameter_um': 65.0}, 'H1': {'depth_um': 0.654, 'diameter_um': 31.0}, 'H2': {'depth_um': 0.654, 'diameter_um': 37.0}, 'H3': {'depth_um': 0.654, 'diameter_um': 44.0}, 'H4': {'depth_um': 0.654, 'diameter_um': 53.0}, 'H5': {'depth_um': 0.654, 'diameter_um': 68.0}}
SPECTRAL_TYPE_TEMPERATURE_K = {'O5V': 40000.0, 'B0V': 30000.0, 'B5V': 15000.0, 'A0V': 9500.0, 'A5V': 8200.0, 'F0V': 7300.0, 'F5V': 6500.0, 'G0V': 5950.0, 'G2V': 5770.0, 'G5V': 5600.0, 'K0V': 5250.0, 'K5V': 4400.0, 'M0V': 3850.0, 'M2V': 3500.0, 'M5V': 3150.0}

def _disc(dim: int, diameter: float, obstruction: float=0.0) -> np.ndarray:
    center = (dim - 1) / 2.0
    yy, xx = np.indices((dim, dim), dtype=float)
    radius = np.hypot(xx - center, yy - center)
    outer = radius <= diameter / 2.0
    if obstruction <= 0:
        return outer.astype(float)
    inner = radius <= diameter * obstruction / 2.0
    return (outer & ~inner).astype(float)

def _rotate_xy(x, y, angle_deg):
    angle = np.deg2rad(angle_deg)
    c = np.cos(angle)
    s = np.sin(angle)
    return (c * x - s * y, s * x + c * y)

def _telescope_pupil(dim: int, diameter: float, geometry: str, rotation_deg: float=0.0, supersample: int=8) -> np.ndarray:
    if geometry == 'ut':
        obstruction = 1100 / 8000
        spider_tilt = 5.5
    elif geometry == 'at':
        obstruction = 0.13 / 1.8
        spider_tilt = 2.5
    else:
        raise ValueError('Telescope pupil must be UT or AT.')
    spider_width = 0.008 * diameter
    ss = int(supersample)
    q = (np.arange(dim * ss, dtype=float) + 0.5) / ss - dim / 2.0
    y, x = np.meshgrid(q, q, indexing='ij')
    radius = np.hypot(x, y)
    outer_radius = diameter / 2.0
    inner_radius = obstruction * outer_radius
    pupil = (radius <= outer_radius) & (radius >= inner_radius)
    blocked = np.zeros_like(pupil, dtype=bool)
    outer_points = ((0.0, +outer_radius), (0.0, -outer_radius), (+outer_radius, 0.0), (-outer_radius, 0.0))
    spider_tilts = (-spider_tilt, -spider_tilt, +spider_tilt, +spider_tilt)
    global_rotation = 45.0 + rotation_deg
    for (outer_x, outer_y), tilt in zip(outer_points, spider_tilts):
        vector_x = -outer_x
        vector_y = -outer_y
        vector_x, vector_y = _rotate_xy(vector_x, vector_y, tilt)
        inner_x = outer_x + vector_x
        inner_y = outer_y + vector_y
        inner_x, inner_y = _rotate_xy(inner_x, inner_y, global_rotation)
        outer_x_rot, outer_y_rot = _rotate_xy(outer_x, outer_y, global_rotation)
        dx = outer_x_rot - inner_x
        dy = outer_y_rot - inner_y
        length = np.hypot(dx, dy)
        ux = dx / length
        uy = dy / length
        px = x - inner_x
        py = y - inner_y
        along = px * ux + py * uy
        across = -px * uy + py * ux
        blocked |= (along >= 0.0) & (along <= length) & (np.abs(across) <= spider_width / 2.0)
    pupil &= ~blocked
    pupil = pupil.reshape(dim, ss, dim, ss).mean(axis=(1, 3))
    return pupil

def _custom_pupil(dim: int, diameter: float, pupil: dict) -> np.ndarray:
    obstruction = float(pupil.get('central_obstruction', 0.0))
    if not 0.0 <= obstruction < 1.0:
        raise ValueError('pupil.central_obstruction must lie in [0, 1).')
    result = _disc(dim, diameter, obstruction)
    spiders = pupil.get('spiders')
    if spiders is None:
        return result
    if 'orientation_deg' in spiders:
        raise ValueError('pupil.spiders.orientation_deg is no longer supported; use pupil.rotation_deg to rotate the complete pupil.')
    count = int(spiders.get('count', 0))
    thickness = float(spiders.get('thickness', 0.0))
    if count < 0:
        raise ValueError('pupil.spiders.count must be non-negative.')
    if thickness < 0:
        raise ValueError('pupil.spiders.thickness must be non-negative.')
    if count == 0 or thickness == 0:
        return result
    center = (dim - 1) / 2.0
    y, x = np.indices((dim, dim), dtype=float)
    x -= center
    y -= center
    inner_radius = obstruction * diameter / 2
    outer_radius = diameter / 2
    half_width = thickness * diameter / 2
    spider_mask = np.zeros((dim, dim), dtype=bool)
    for index in range(count):
        angle = np.deg2rad(index * 360.0 / count)
        along = x * np.cos(angle) + y * np.sin(angle)
        across = -x * np.sin(angle) + y * np.cos(angle)
        spider_mask |= (along >= inner_radius) & (along <= outer_radius) & (np.abs(across) <= half_width)
    result[spider_mask] = 0.0
    return result

def make_pupil(config: dict) -> np.ndarray:
    pupil = config['pupil']
    dim = int(pupil['array_size'])
    diameter = float(pupil['pixels_across_pupil'])
    geometry_value = pupil.get('geometry')
    rotation = float(pupil.get('rotation_deg', 0.0))
    if geometry_value is None or not str(geometry_value).strip():
        result = _custom_pupil(dim, diameter, pupil)
    else:
        geometry = str(geometry_value).strip().lower()
        if geometry == 'solarstein':
            result = _disc(dim, diameter, 1100 / 8000)
        elif geometry in {'disc', 'disk'}:
            result = _disc(dim, diameter)
        elif geometry in {'ut', 'at'}:
            return _telescope_pupil(dim, diameter, geometry, rotation_deg=rotation, supersample=8)
        else:
            raise ValueError("pupil.geometry must be 'solarstein', 'disc', 'UT', 'AT', or omitted for a custom pupil.")
    if rotation:
        center = ((dim - 1) / 2, (dim - 1) / 2)
        result = np.clip(_rotate_about(result, rotation, center), 0.0, 1.0)
    return result

def _planck(wavelength_m: np.ndarray, temperature_k: float) -> np.ndarray:
    h = 6.62607015e-34
    c = 299792458.0
    k = 1.380649e-23
    exponent = h * c / (wavelength_m * k * temperature_k)
    return 2 * h * c ** 2 / wavelength_m ** 5 / np.expm1(exponent)

def resolve_source_profile(config: dict) -> tuple[str, dict, float, str | None]:
    """Resolve the selected profile and its effective blackbody temperature.

    A non-null ``temperature_K`` always wins. Otherwise ``spectral_type`` is
    looked up in ``SPECTRAL_TYPE_TEMPERATURE_K``.
    """
    source = config['source']
    profile_name = str(source['profile'])
    try:
        profile = source['profiles'][profile_name]
    except KeyError as error:
        raise ValueError(f'Unknown source profile {profile_name!r}.') from error
    temperature = profile.get('temperature_K')
    spectral_type = profile.get('spectral_type')
    print(f"temperature={temperature}, spectral_type={spectral_type}")
    
    if temperature is not None:
        temperature = float(temperature)
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError('source profile temperature_K must be finite and positive.')
        return (profile_name, profile, temperature, spectral_type)
    if spectral_type is None or not str(spectral_type).strip():
        raise ValueError(f'Source profile {profile_name!r} must provide temperature_K or spectral_type.')
    spectral_type = str(spectral_type).strip().upper().replace(' ', '')
    try:
        temperature = SPECTRAL_TYPE_TEMPERATURE_K[spectral_type]
    except KeyError as error:
        temperature = 1500
        #raise ValueError(f'Unknown spectral type {spectral_type!r}; supported types are {', '.join(SPECTRAL_TYPE_TEMPERATURE_K)}.') from error
    return (profile_name, profile, temperature, spectral_type)

def make_spectrum(config: dict) -> tuple[np.ndarray, np.ndarray]:
    _, _, temperature_k, _ = resolve_source_profile(config)
    spectrum = config['spectrum']
    wavelength_min_m = float(spectrum['wavelength_min_m'])
    wavelength_max_m = float(spectrum['wavelength_max_m'])
    if wavelength_max_m <= wavelength_min_m:
        raise ValueError('spectrum.wavelength_max_m must exceed wavelength_min_m.')
    wavelengths = np.linspace(wavelength_min_m, wavelength_max_m, int(spectrum['samples']))
    mode = str(spectrum.get('mode', 'blackbody')).lower()
    if mode == 'flat':
        weights = np.ones_like(wavelengths)
    elif mode == 'blackbody':
        weights = _planck(wavelengths, temperature_k)
        if str(spectrum.get('weighting', 'photon')).lower() == 'photon':
            weights *= wavelengths
    else:
        raise ValueError("spectrum.mode must be 'blackbody' or 'flat'.")
    weights /= weights.sum()
    bandwidth_nm = (wavelength_max_m - wavelength_min_m) * 1000000000.0
    return (wavelengths, weights * bandwidth_nm)

def _linear_interpolator(data_path: Path):
    table = np.loadtxt(_StringIO(_embedded_material) if str(data_path) == 'embedded:material_table' else data_path, skiprows=2, usecols=(0, 1))
    return interp1d(table[:, 0], table[:, 1], kind='linear', bounds_error=False, fill_value='extrapolate')

def mask_parameters(config: dict, wavelength_m: float, material_data_path: Path) -> tuple[float, float]:
    mask = dict(config['phase_mask'])
    mask_name = mask.get('name')
    if mask_name is not None and str(mask_name).strip():
        mask_name = str(mask_name).strip().upper()
        if mask_name not in PHASE_MASK_PRESETS:
            raise ValueError(f'Unknown phase-mask name {mask_name!r}; use H1-H5, J1-J5, or null for custom physical parameters.')
        mask.update(PHASE_MASK_PRESETS[mask_name])
        mask['phase_model'] = 'physical_depth'
        mask['diameter_model'] = 'physical'
        mask.setdefault('material', 'N_1405')
    theta_model = str(mask.get('phase_model', 'physical_depth')).lower()
    if theta_model == 'constant':
        theta = float(mask['phase_shift_rad'])
    elif theta_model == 'physical_depth':
        wavelength_um = wavelength_m * 1000000.0
        refractive_index = float(_linear_interpolator(material_data_path)(wavelength_um * 1000.0))
        theta = 2 * np.pi / wavelength_um * float(mask['depth_um']) * (refractive_index - 1)
    else:
        raise ValueError("phase_mask.phase_model must be 'constant' or 'physical_depth'.")
    diameter_model = str(mask.get('diameter_model', 'physical')).lower()
    if diameter_model == 'physical':
        diameter_lambda_d = float(mask['diameter_um']) * 1e-06 / (1.22 * float(config['optics']['f_number']) * wavelength_m)
    elif diameter_model == 'lambda_over_d':
        diameter_lambda_d = float(mask['diameter_lambda_over_d'])
    else:
        raise ValueError("phase_mask.diameter_model must be 'physical' or 'lambda_over_d'.")
    return (theta, diameter_lambda_d)

def zwfs_field(amplitude: np.ndarray, theta: float, diameter_lambda_d: float, focal_array_size: int=300, phase: np.ndarray | None=None, strehl: float=1.0) -> np.ndarray:
    """Return the post-phase-mask pupil field using BaldrApp centering."""
    original_size = amplitude.shape[0]
    if amplitude.shape != (original_size, original_size):
        raise ValueError('The input amplitude must be square.')
    entrance = amplitude.astype(complex)
    if phase is not None:
        entrance *= np.exp(1j * np.asarray(phase, dtype=float))
    if original_size % 2 == 0:
        work = np.pad(entrance, ((0, 1), (0, 1)))
    else:
        work = entrance
    size = work.shape[0]
    focal_size = max(int(focal_array_size), size)
    if focal_size % 2 == 0:
        focal_size += 1
    pad = (focal_size - size) // 2
    pupil_field = np.zeros((focal_size, focal_size), dtype=complex)
    pupil_field[pad:pad + size, pad:pad + size] = work
    focal_field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil_field), norm='ortho'))
    support = np.zeros((focal_size, focal_size), dtype=bool)
    support[pad:pad + size, pad:pad + size] = np.abs(work) > 0
    yy, xx = np.indices((focal_size, focal_size))
    center = (focal_size - 1) / 2.0
    radius = np.hypot(yy - center, xx - center)
    pupil_diameter_pixels = 2.0 * radius[support].max()
    pixels_per_lambda_d = focal_size / pupil_diameter_pixels
    phase_disc = radius <= 0.5 * diameter_lambda_d * pixels_per_lambda_d
    reference_wave = np.sqrt(strehl) * np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(phase_disc * focal_field), norm='ortho'))
    output = pupil_field + (np.exp(1j * theta) - 1.0) * reference_wave
    output = output[pad:pad + size, pad:pad + size]
    return output[:original_size, :original_size]

def _coordinate_grid(shape: tuple[int, int], dx: float, dy: float | None=None):
    if dy is None:
        dy = dx
    ny, nx = shape
    x = (np.arange(nx) - nx // 2) * dx
    y = (np.arange(ny) - ny // 2) * dy
    return np.meshgrid(x, y)

def _angular_spectrum(field, wavelength, dx, z, dy=None):
    if dy is None:
        dy = dx
    ny, nx = field.shape
    fx = np.fft.fftfreq(nx, d=dx)
    fy = np.fft.fftfreq(ny, d=dy)
    fx_grid, fy_grid = np.meshgrid(fx, fy)
    argument = 1.0 - (wavelength * fx_grid) ** 2 - (wavelength * fy_grid) ** 2
    valid = argument >= 0
    transfer = np.zeros(field.shape, dtype=complex)
    transfer[valid] = np.exp(1j * 2 * np.pi / wavelength * z * np.sqrt(argument[valid]))
    return np.fft.ifft2(np.fft.fft2(field) * transfer)

def _thin_lens(field, wavelength, dx, focal_length):
    x, y = _coordinate_grid(field.shape, dx)
    return field * np.exp(-1j * np.pi * (x ** 2 + y ** 2) / (wavelength * focal_length))

def _one_step_fresnel(field, wavelength, dx, z, dy=None):
    if dy is None:
        dy = dx
    ny, nx = field.shape
    k = 2 * np.pi / wavelength
    x1, y1 = _coordinate_grid(field.shape, dx, dy)
    dx_out = wavelength * abs(z) / (nx * dx)
    dy_out = wavelength * abs(z) / (ny * dy)
    x2, y2 = _coordinate_grid(field.shape, dx_out, dy_out)
    quad_in = np.exp(1j * k * (x1 ** 2 + y1 ** 2) / (2 * z))
    quad_out = np.exp(1j * k * (x2 ** 2 + y2 ** 2) / (2 * z))
    transformed = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field * quad_in)))
    output = dx * dy / (1j * wavelength * z) * quad_out * transformed
    return (output, dx_out, dy_out)

def derive_relay(config: dict) -> dict:
    relay = dict(config['relay'])
    diameter = relay['entrance_diameter_m'] * relay['collimator_focal_length_m'] / relay['oap_focal_length_m']
    magnification = relay['detector_pupil_diameter_m'] / diameter
    object_nominal = relay['imaging_focal_length_m'] * (1 + 1 / magnification)
    image_nominal = relay['imaging_focal_length_m'] * (1 + magnification)
    mirror_to_lens = object_nominal - relay['distance_to_mirror_m']
    object_distance = relay['distance_to_mirror_m'] + mirror_to_lens
    image_distance = 1 / (1 / relay['imaging_focal_length_m'] - 1 / object_distance)
    relay.update(physical_pupil_diameter_m=diameter, mirror_to_lens_m=mirror_to_lens, focus_to_detector_m=image_distance - relay['imaging_focal_length_m'] + relay.get('pupil_misconjugation_m', 0.0), nominal_image_distance_m=image_nominal)
    return relay

def flat_dm_opd(config: dict) -> np.ndarray:
    """Reproduce BaldrApp's nominal 0.5-command BMC flat surface."""
    pupil = config['pupil']
    dm = config['flat_dm']
    size = int(pupil['array_size'])
    telescope_diameter = float(pupil['physical_diameter_m'])
    padding_factor = size / float(pupil['pixels_across_pupil'])
    lower = -(telescope_diameter * padding_factor) // 2
    upper = telescope_diameter * padding_factor // 2
    coordinate = np.linspace(lower, upper, size)
    x_grid, y_grid = np.meshgrid(coordinate, coordinate)
    actuator_scale = 12 / 10 * (telescope_diameter / 2) / (12 / 2 - 0.5)
    actuator_coordinates = []
    for row in range(12):
        for column in range(12):
            if row in {0, 11} and column in {0, 11}:
                continue
            actuator_coordinates.append(((column - 5.5) * actuator_scale + coordinate.mean(), (row - 5.5) * actuator_scale + coordinate.mean()))
    sigma = float(dm['actuator_coupling_factor']) * actuator_scale
    command_opd = float(dm['flat_command']) * float(dm['opd_per_command_m'])
    result = np.zeros((size, size), dtype=float)
    for x0, y0 in actuator_coordinates:
        result += command_opd * np.exp(-((x_grid - x0) ** 2 + (y_grid - y0) ** 2) / sigma ** 2)
    return result

def relay_intensity(field, wavelength: float, config: dict) -> np.ndarray:
    pupil = config['pupil']
    relay = derive_relay(config)
    dx = relay['physical_pupil_diameter_m'] / float(pupil['pixels_across_pupil'])
    x, y = _coordinate_grid(field.shape, dx)
    mirror = _angular_spectrum(field, wavelength, dx, relay['distance_to_mirror_m'])
    full_mirror = np.hypot(x, y) <= relay['mirror_diameter_m'] / 2
    angle = relay.get('edge_angle_rad', 0.0)
    rotated_x = x * np.cos(angle) + y * np.sin(angle)
    mirror *= full_mirror & (rotated_x >= relay.get('edge_offset_m', 0.0))
    at_lens = _angular_spectrum(mirror, wavelength, dx, relay['mirror_to_lens_m'])
    after_lens = _thin_lens(at_lens, wavelength, dx, relay['imaging_focal_length_m'])
    at_stop, dx_stop, dy_stop = _one_step_fresnel(after_lens, wavelength, dx, relay['imaging_focal_length_m'])
    xs, ys = _coordinate_grid(at_stop.shape, dx_stop, dy_stop)
    stop_radius = np.hypot(xs - relay.get('cold_stop_x_offset_m', 0.0), ys - relay.get('cold_stop_y_offset_m', 0.0))
    after_stop = at_stop * (stop_radius <= relay['cold_stop_diameter_m'] / 2)
    detector_field, dx_detector, dy_detector = _one_step_fresnel(after_stop, wavelength, dx_stop, relay['focus_to_detector_m'], dy_stop)
    return np.abs(detector_field) ** 2 * dx_detector * dy_detector / dx ** 2

def _bin(image: np.ndarray, factor: int) -> np.ndarray:
    if image.shape[0] % factor or image.shape[1] % factor:
        raise ValueError('Image dimensions must be divisible by detector binning.')
    return image.reshape(image.shape[0] // factor, factor, image.shape[1] // factor, factor).sum(axis=(1, 3))

def _crop(image: np.ndarray, shape) -> np.ndarray:
    if shape is None:
        return image
    height, width = map(int, shape)
    y0 = (image.shape[0] - height) // 2
    x0 = (image.shape[1] - width) // 2
    if y0 < 0 or x0 < 0:
        raise ValueError('Detector crop is larger than the binned image.')
    return image[y0:y0 + height, x0:x0 + width]

def generate_clear_reference(config: dict, pupil_amplitude: np.ndarray | None=None) -> tuple[np.ndarray, np.ndarray]:
    """Generate only the clear reference, avoiding phase-mask propagation."""
    pupil = make_pupil(config)
    if pupil_amplitude is None:
        pupil_amplitude = pupil
    else:
        pupil_amplitude = np.asarray(pupil_amplitude, dtype=float)
        if pupil_amplitude.shape != pupil.shape:
            raise ValueError(f'pupil_amplitude must have shape {pupil.shape}, got {pupil_amplitude.shape}.')
        if not np.all(np.isfinite(pupil_amplitude)):
            raise ValueError('pupil_amplitude contains non-finite values.')
        if np.any(pupil_amplitude < 0):
            raise ValueError('pupil_amplitude must be non-negative.')
    _, profile, _, _ = resolve_source_profile(config)
    amplitude = np.sqrt(float(profile['photons_per_second_per_pixel_per_nm'])) * pupil_amplitude
    wavelengths, weights_nm = make_spectrum(config)
    dm_opd = flat_dm_opd(config)
    clear_rate = np.zeros_like(pupil)
    for wavelength, weight_nm in zip(wavelengths, weights_nm):
        entrance = amplitude * np.exp(1j * 2 * np.pi / float(wavelength) * dm_opd)
        clear_rate += weight_nm * relay_intensity(entrance, float(wavelength), config)
    detector = config['detector']
    scale = float(detector['quantum_efficiency']) * float(detector['exposure_s'])
    clear = _bin(clear_rate, int(detector['binning'])) * scale
    return (_crop(clear, detector.get('crop')), wavelengths)

def generate_references(config: dict, material_data_path: Path, pupil_amplitude: np.ndarray | None=None):
    """Generate clear and ZWFS references.

    ``pupil_amplitude`` is an optional dimensionless, high-resolution entrance
    pupil amplitude sampled on the same grid as ``make_pupil(config)``.  When
    omitted, the configured analytic pupil is used exactly as before.
    """
    pupil = make_pupil(config)
    if pupil_amplitude is None:
        pupil_amplitude = pupil
    else:
        pupil_amplitude = np.asarray(pupil_amplitude, dtype=float)
        if pupil_amplitude.shape != pupil.shape:
            raise ValueError(f'pupil_amplitude must have shape {pupil.shape}, got {pupil_amplitude.shape}.')
        if not np.all(np.isfinite(pupil_amplitude)):
            raise ValueError('pupil_amplitude contains non-finite values.')
        if np.any(pupil_amplitude < 0):
            raise ValueError('pupil_amplitude must be non-negative.')
    _, profile, _, _ = resolve_source_profile(config)
    amplitude = np.sqrt(float(profile['photons_per_second_per_pixel_per_nm'])) * pupil_amplitude
    wavelengths, weights_nm = make_spectrum(config)
    dm_opd = flat_dm_opd(config)
    clear_rate = np.zeros_like(pupil)
    masked_rate = np.zeros_like(pupil)
    strehl = float(config.get('optics', {}).get('strehl', config.get('strehl', 1.0)))
    if not np.isfinite(strehl) or not 0.0 <= strehl <= 1.0:
        raise ValueError('optics.strehl must be finite and lie in [0, 1].')
    for wavelength, weight_nm in zip(wavelengths, weights_nm):
        theta, diameter = mask_parameters(config, float(wavelength), material_data_path)
        clear_rate += weight_nm * relay_intensity(zwfs_field(amplitude, 0.0, diameter, phase=2 * np.pi / float(wavelength) * dm_opd), float(wavelength), config)
        masked_rate += weight_nm * relay_intensity(zwfs_field(amplitude, theta, diameter, phase=2 * np.pi / float(wavelength) * dm_opd, strehl=strehl), float(wavelength), config)
    detector = config['detector']
    scale = float(detector['quantum_efficiency']) * float(detector['exposure_s'])
    clear = _bin(clear_rate, int(detector['binning'])) * scale
    masked = _bin(masked_rate, int(detector['binning'])) * scale
    crop = detector.get('crop')
    return (_crop(clear, crop), _crop(masked, crop), wavelengths)

def load_config(path: str | Path) -> dict:
    if str(path) == 'embedded:reference_config.json':
        return json.loads(_embedded_reference)
    with Path(path).open() as stream:
        return json.load(stream)


def _rotate_about(array, angle_deg, center):
    """Restore the helper called by non-UT/AT rotated pupils in the source model."""
    angle = -np.deg2rad(angle_deg)
    yy, xx = np.indices(array.shape, dtype=float)
    xp = (xx-center[0])*np.cos(angle)+(yy-center[1])*np.sin(angle)+center[0]
    yp = -(xx-center[0])*np.sin(angle)+(yy-center[1])*np.cos(angle)+center[1]
    return ndimage.map_coordinates(array, [yp, xp], mode='constant', cval=0., order=3)


# Hardware

from pathlib import Path

MDS_HOST_ONSKY = 'mimir'
MDS_HOST_SIMULATOR = '127.0.0.1'


class Device:
    """Same channel map and notification as DM_shm_ctrl, without flat-file helpers."""
    def __init__(self, beam_id, main_chn=2, shm_directory='/dev/shm'):
        from xaosim.shmlib import shm
        directory = Path(shm_directory)
        # Explicit channel suffixes avoid reindexing when a channel file is missing.
        files = sorted(directory.glob(f'dm{beam_id}disp*.im.shm'))
        self.shms, self.shm0, self.main_chn = [], None, main_chn
        suffixes = [int(p.name.split('disp')[1].split('.')[0]) for p in files]
        if suffixes != list(range(len(files))) or main_chn >= len(files):
            raise RuntimeError('Missing/nonconsecutive DM channels; start the DM server')
        combined = directory/f'dm{beam_id}.im.shm'
        if not combined.is_file():
            raise FileNotFoundError(combined)
        try:
            for path in files:
                self.shms.append(shm(str(path), nosem=False))
            self.shm0 = shm(str(combined), nosem=False)
        except BaseException:
            self.close()
            raise

    def close(self, erase_file=False):
        # Never permit deletion of producer-owned files.
        errors = []
        for stream in self.shms + ([self.shm0] if self.shm0 is not None else []):
            try:
                stream.close(erase_file=False)
            except Exception as exc:
                errors.append(exc)
        self.shms = []
        self.shm0 = None
        if errors:
            raise RuntimeError(f'Error closing SHM attachments: {errors}')


# Calibration

#!/usr/bin/env python3
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import signal
import time

import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares

ACTUATORS = np.ones((12, 12), bool)
ACTUATORS[0, 0] = ACTUATORS[0, -1] = False
ACTUATORS[-1, 0] = ACTUATORS[-1, -1] = False


def dm_map(vector):
    vector = np.asarray(vector, float)
    if vector.shape != (140,) or not np.isfinite(vector).all():
        raise ValueError('DM vector must contain 140 finite values')
    result = np.zeros((12, 12))
    result[ACTUATORS] = vector
    return result


def projection(shape, xy):
    """Bilinear samples, x=column, y=row, C-order image flattening.

    Unlike the original helper, never extrapolate outside the camera image.
    Invalid rows are zero and MUST be excluded using the returned validity mask.
    """
    h, w = shape
    xy = np.asarray(xy, float)
    good = np.isfinite(xy).all(axis=1)
    good &= (xy[:, 0] >= 0) & (xy[:, 0] <= w-1)
    good &= (xy[:, 1] >= 0) & (xy[:, 1] <= h-1)
    matrix = np.zeros((len(xy), h*w))
    for i in np.flatnonzero(good):
        x, y = xy[i]
        x0, y0 = min(int(x), w-2), min(int(y), h-2)
        dx, dy = x-x0, y-y0
        matrix[i, [y0*w+x0, y0*w+x0+1, (y0+1)*w+x0, (y0+1)*w+x0+1]] = (
            (1-dx)*(1-dy), dx*(1-dy), (1-dx)*dy, dx*dy)
    return matrix, good


def halo_pupil(shape, cx, cy, radius, margin):
    """Freeze a widened inside region from clear geometry, never mask-IN intensity."""
    if not np.isfinite(margin) or margin < 0:
        raise ValueError('halo_margin_px must be finite and nonnegative')
    h, w = shape
    expanded = radius+margin
    if min(cx, cy, w-1-cx, h-1-cy) <= expanded:
        raise RuntimeError('Expanded halo boundary is truncated by camera crop; reduce halo_margin_px')
    yy, xx = np.indices(shape)
    return np.hypot(xx-cx, yy-cy) <= expanded


def transform(image, params):
    angle, shift_x, shift_y, scale = params
    angle = np.deg2rad(angle)
    matrix = np.array([[np.cos(angle), -np.sin(angle)],
                       [np.sin(angle), np.cos(angle)]]) / scale
    center = (np.array(image.shape)-1)/2
    return ndimage.affine_transform(image, matrix,
        offset=center-matrix @ (center+[shift_y, shift_x]),
        order=1, mode='constant', cval=0., prefilter=False)


def fit_theory(clear, theory_clear, theory_masked, support, good, estimate, max_error):
    """Fit geometry using clear only; use its flux scale for BOTH model images."""
    measured_geometry = estimate(clear)
    theory_geometry = estimate(theory_clear)
    start = np.array([0., measured_geometry['center_x']-theory_geometry['center_x'],
        measured_geometry['center_y']-theory_geometry['center_y'],
        measured_geometry['radius_pixels']/theory_geometry['radius_pixels']])
    lower = start + [-5., -2., -2., -.08*start[3]]
    upper = start + [5., 2., 2., .08*start[3]]
    measured = clear/np.mean(clear[support])
    def residual(params):
        trial = transform(theory_clear, params)
        scale = np.mean(trial[support])
        if scale <= 0:
            return np.full(good.sum(), 1e6)
        return (trial/scale-measured)[good]
    fit = least_squares(residual, start, bounds=(lower, upper), max_nfev=200)
    error = float(np.sqrt(np.mean(residual(fit.x)**2)))
    if not fit.success or np.any(fit.active_mask) or error > max_error:
        raise RuntimeError(f'Theory alignment failed or reached bounds; clear RMSE={error:.4g}')
    aligned_clear = transform(theory_clear, fit.x)
    scale = np.mean(aligned_clear[support])
    return aligned_clear/scale, transform(theory_masked, fit.x)/scale, fit.x, error


def rpc(address, command, timeout_ms=3000):
    """One REQ socket per command, including recovery after ambiguous timeouts."""
    import zmq
    with zmq.Context() as context:
        with context.socket(zmq.REQ) as socket:
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
            socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
            socket.connect(address)
            socket.send_string(command)
            reply = socket.recv_string().strip()
    if not reply or reply.upper().startswith(('ERR', 'NACK', 'NOK', 'FAIL')) or 'not connected' in reply.lower():
        raise RuntimeError(f'{command}: {reply}')
    return reply


def position(call, axis):
    value = float(call(f'read {axis}'))
    if not np.isfinite(value):
        raise RuntimeError('Nonfinite mask position')
    return value


def wait_position(call, axis, target, timeout, tolerance=.1):
    deadline = time.monotonic()+timeout
    while abs(position(call, axis)-target) > tolerance:
        if time.monotonic() > deadline:
            raise TimeoutError(f'{axis} did not reach {target}')
        time.sleep(.05)


@contextmanager
def mask_out(call, axis, offset, settle, timeout):
    original = position(call, axis)
    # Enter the recovery block BEFORE sending a move: a timeout may mean it moved.
    try:
        call(f'moverel {axis} {offset:.9g}')
        wait_position(call, axis, original+offset, timeout)
        time.sleep(settle)
        yield original
    finally:
        try:
            # Wait for a possibly timed-out motion to stop before recovering.
            deadline = time.monotonic()+timeout
            previous = position(call, axis)
            stable = 0
            while stable < 3:
                time.sleep(.2)
                current = position(call, axis)
                stable = stable+1 if abs(current-previous) < .01 else 0
                previous = current
                if time.monotonic() > deadline:
                    raise TimeoutError('Stage did not stop')
            difference = original-current
            if abs(difference) > .1:
                call(f'moverel {axis} {difference:.9g}')
            wait_position(call, axis, original, timeout)
            time.sleep(settle)
        except BaseException as exc:
            raise RuntimeError(f'MASK RESTORATION UNVERIFIED: {axis}, original={original}') from exc


class Camera:
    def __init__(self, path):
        from xaosim.shmlib import shm
        if not Path(path).is_file():
            raise FileNotFoundError(path)
        self.stream = shm(str(path), nosem=True)
        self.last = None

    def read(self):
        counter = int(self.stream.get_counter())
        if counter == self.last:
            return None
        if self.last is not None and counter < self.last:
            raise RuntimeError('Camera counter reset')
        frame = self.stream.get_data().copy()
        if int(self.stream.get_counter()) != counter:
            return None
        self.last = counter
        return counter, frame

    def close(self):
        self.stream.close(erase_file=False)


def fresh_frames(camera, count, timeout):
    camera.read()  # Discard latest exposure after settling.
    deadline = time.monotonic()+timeout
    frames, counters = [], []
    while len(frames) < count:
        if time.monotonic() > deadline:
            raise TimeoutError(f'Only {len(frames)}/{count} fresh frames')
        sample = camera.read()
        if sample is None:
            time.sleep(.005)
            continue
        counter, frame = sample
        frame = np.asarray(frame, float)
        if frame.ndim != 2 or (frames and frame.shape != frames[0].shape):
            raise ValueError('Camera frame shape changed or is not 2D')
        if counters and counter <= counters[-1]:
            raise RuntimeError('Repeated or reset camera counter')
        frames.append(frame.copy())
        counters.append(counter)
    return np.array(frames), np.array(counters)


class DM:
    """Own one channel temporarily; never zero other channels or activate a flat."""
    def __init__(self, device, channel, limits, max_offset):
        self.device, self.channel = device, channel
        self.limits, self.max_offset = limits, max_offset
        if channel == 0 or channel >= len(device.shms):
            raise ValueError('Choose an existing non-flat DM channel')
        self.initial = np.array([s.get_data().copy() for s in device.shms])
        if self.initial.shape[1:] != (12, 12) or not np.isfinite(self.initial).all():
            raise ValueError('Expected finite 12x12 DM channel maps')
        self.expected = self.initial[channel].copy()
        self.changed = False
        self.validate(np.zeros(140))

    def check_owner(self):
        for i, stream in enumerate(self.device.shms):
            expected = self.expected if i == self.channel else self.initial[i]
            if not np.allclose(stream.get_data(), expected, rtol=0, atol=1e-7):
                raise RuntimeError(f'DM channel {i} changed externally; stop competing controllers')

    def validate(self, delta):
        offset = dm_map(delta)
        combined = self.initial.sum(axis=0)+offset
        if np.max(np.abs(delta)) > self.max_offset+1e-12:
            raise ValueError('Cumulative DM offset exceeds configured limit')
        if np.any(combined[ACTUATORS] < self.limits[0]) or np.any(combined[ACTUATORS] > self.limits[1]):
            raise ValueError('Summed DM command exceeds configured absolute limits')
        return self.initial[self.channel]+offset

    def write(self, delta):
        command = self.validate(delta)
        self.check_owner()
        self.changed = True  # Even a failed write can have changed the channel.
        self.expected = command.astype(self.initial.dtype)
        self.device.shms[self.channel].set_data(self.expected)
        self.device.shm0.post_sems(1)

    def restore(self):
        if not self.changed:
            return
        # Do not overwrite a competing writer on our own channel.
        current = self.device.shms[self.channel].get_data()
        if np.allclose(current, self.initial[self.channel], rtol=0, atol=1e-7):
            self.expected = self.initial[self.channel].copy()
            return
        if not np.allclose(current, self.expected, rtol=0, atol=1e-7):
            raise RuntimeError('DM RESTORATION UNVERIFIED: owned channel changed externally')
        self.device.shms[self.channel].set_data(self.initial[self.channel])
        self.device.shm0.post_sems(1)
        if not np.allclose(self.device.shms[self.channel].get_data(), self.initial[self.channel], rtol=0, atol=1e-7):
            raise RuntimeError('DM RESTORATION UNVERIFIED: readback mismatch')
        self.expected = self.initial[self.channel].copy()


def register_dm(dm, capture, support, good, cfg, registration):
    corners = registration.get_inner_square_indices(outer_size=12, inner_offset=4)
    images = []
    def poke_image(act):
        pair_means, pair_errors = [], []
        for sign in (1., -1.):
            command = np.zeros(140)
            command[act] = sign*cfg['poke_amplitude']
            dm.write(command)
            cube = capture(f'poke_{act}_{sign:g}', cfg['poke_frames'])
            pair_means.append(cube.mean(axis=0))
            pair_errors.append(cube.var(axis=0, ddof=1)/len(cube))
        delta = abs(pair_means[0]-pair_means[1])*support
        noise = np.sqrt(pair_errors[0]+pair_errors[1])
        peak = np.unravel_index(np.argmax(delta), delta.shape)
        if delta[peak] <= 5*max(noise[peak], 1e-12):
            raise RuntimeError(f'Actuator {act} poke has insufficient SNR')
        return delta
    try:
        for act in corners:
            images.append(poke_image(act))
        fit = registration.calibrate_transform_between_DM_and_image(corners, images, debug=False)
        xy = np.asarray(fit['actuator_coord_list_pixel_space'])
        observed = np.asarray(fit['img_corners'])
        errors = np.linalg.norm(xy[corners]-observed, axis=1)
        # Independent fifth actuator guards against a plausible but incorrect four-point fit.
        preliminary, in_frame = projection(support.shape, xy)
        interior = ndimage.distance_transform_edt(support) >= 2.5
        candidates = in_frame & ((preliminary @ interior.ravel().astype(float)) > .99)
        candidates[corners] = False
        if not candidates.any():
            raise RuntimeError('No illuminated independent registration actuator')
        # Choose a well-lit point away from the fitted corners, avoiding the obstruction.
        distances = np.linalg.norm(xy[:, None, :]-xy[corners][None, :, :], axis=2).min(axis=1)
        holdout = int(np.argmax(np.where(candidates, distances, -1.)))
        test = poke_image(holdout)
        peak = np.unravel_index(np.argmax(test), test.shape)
        local_fit = registration.interpolate_and_fit_gaussian(peak, 5, test, factor=5)
        measured = np.array([local_fit['x0'], local_fit['y0']])
        holdout_error = float(np.linalg.norm(xy[holdout]-measured))
        if (not np.isfinite(xy).all() or not np.isfinite(errors).all()
                or not np.isfinite(holdout_error)
                or max(errors.max(), holdout_error) > cfg['registration_tolerance_px']):
            raise RuntimeError('DM registration failed corner/independent-actuator checks')
        matrix, inside = projection(support.shape, xy)
        active = inside & ((matrix @ support.ravel().astype(float)) > .99)
        active &= (matrix @ good.ravel().astype(float)) > .999999
        if active.sum() < 12:
            raise RuntimeError('Too few illuminated, reliable actuators')
        return matrix, active, xy, np.array(images), errors, holdout_error
    finally:
        dm.write(np.zeros(140))


def standard_error(values):
    return float(np.std(values, ddof=1)/np.sqrt(len(values)))


class Metrics:
    def __init__(self, clear_cube, pupil, support, good, matrix, active, target, cfg):
        self.clear = clear_cube.mean(axis=0)
        self.clear_count = len(clear_cube)
        self.pupil, self.good = pupil, good
        self.matrix, self.active, self.target = matrix, active, target
        self.norm = float(self.clear[support].mean())
        self.cfg = cfg
        self.initial_excess = None
        self.initial_flux = None

    def measure(self, cube):
        halo = measure_halo(cube, self.clear, self.pupil, self.cfg['_effective_noise_e'],
            self.clear_count, 0., weights=self.good.astype(float))
        valid = halo['valid']
        if valid.sum() < max(3, .8*len(cube)):
            raise RuntimeError('Fewer than 80% valid halo frames')
        diagnostic = reference_diagnostics(halo)
        data = cube[valid]
        projected = (data.reshape(len(data), -1)/self.norm) @ self.matrix.T
        residual = projected[:, self.active]-self.target[self.active]
        frame_rmse = np.sqrt(np.mean(residual**2, axis=1))
        rmse = float(np.sqrt(np.mean(residual.mean(axis=0)**2)))
        flux = float(np.mean(halo['inside_e'][valid]+halo['outside_e'][valid]))
        if self.initial_excess is None:
            if diagnostic['excess'] <= 0 or diagnostic['significance'] <= 5:
                raise RuntimeError('Initial mask-induced halo excess must be positive at >5 sigma')
            self.initial_excess, self.initial_flux = diagnostic['excess'], flux
        if abs(flux/self.initial_flux-1) > self.cfg['max_flux_drift']:
            raise RuntimeError('Flux drift exceeds limit; fixed clear normalization is no longer valid')
        return dict(rmse=rmse, rmse_se=standard_error(frame_rmse),
            halo=diagnostic['excess'], halo_se=diagnostic['fraction_sigma'],
            relative_halo=diagnostic['excess']/self.initial_excess,
            flux=flux, valid_frames=int(valid.sum())), projected.mean(axis=0)


def acceptable(candidate, baseline, cfg):
    """Default: significant halo increase, no significant RMSE increase."""
    r_error = cfg['accept_sigma']*np.hypot(candidate['rmse_se'], baseline['rmse_se'])
    h_error = cfg['accept_sigma']*np.hypot(candidate['halo_se'], baseline['halo_se'])
    if cfg['objective'] == 'rmse':
        return (candidate['rmse'] < baseline['rmse']-max(cfg['min_rmse_improvement'], r_error)
                and candidate['halo'] >= baseline['halo']-h_error)
    return (candidate['halo'] > baseline['halo']+h_error
            and candidate['rmse'] <= baseline['rmse']+r_error)


def optimize(dm, capture, metrics, cfg, history):
    best = np.zeros(140)
    step = cfg['scan_peak']
    def evaluate(command, label):
        dm.write(command)
        score, measured = metrics.measure(capture(label, cfg['frames']))
        history.append(dict(label=label, command=command.tolist(), **score))
        print(f"{label}: RMSE={score['rmse']:.5g}, relative halo={score['relative_halo']:.5g}")
        return score, measured
    for iteration in range(cfg['iterations']):
        baseline, measured = evaluate(best, f'baseline_{iteration}')
        direction = np.zeros(140)
        direction[metrics.active] = measured[metrics.active]-metrics.target[metrics.active]
        direction[metrics.active] -= direction[metrics.active].mean()
        peak = np.max(abs(direction))
        if peak < max(1e-12, cfg['min_rmse_improvement']):
            break
        direction /= peak
        trials = []
        # Alternating signs reduces correlation between sign and slow temporal drift.
        amps = [s*a for a in np.linspace(step/(cfg['scan_points']//2), step, cfg['scan_points']//2) for s in (-1, 1)]
        for index, amplitude in enumerate(amps):
            command = best+amplitude*direction
            try:
                dm.validate(command)
            except ValueError:
                continue
            score, _ = evaluate(command, f'trial_{iteration}_{index}')
            if acceptable(score, baseline, cfg):
                trials.append((score, command))
        # Repeat incumbent AFTER scan to detect drift and compare fresh windows.
        baseline_after, _ = evaluate(best, f'baseline_after_{iteration}')
        trials = [(s, c) for s, c in trials if acceptable(s, baseline_after, cfg)]
        if trials:
            key = (lambda item: item[0]['rmse']) if cfg['objective'] == 'rmse' else (lambda item: -item[0]['halo'])
            _, candidate = min(trials, key=key)
            confirmed, _ = evaluate(candidate, f'confirmation_{iteration}')
            if acceptable(confirmed, baseline, cfg) and acceptable(confirmed, baseline_after, cfg):
                best = candidate
                continue
        dm.write(best)
        step /= 2
        if step < cfg['min_scan_peak']:
            break
    # An independent final A/B comparison rejects accumulated drift or regression.
    initial, _ = evaluate(np.zeros(140), 'final_starting_state')
    final, _ = evaluate(best, 'final_candidate')
    if np.any(best) and not acceptable(final, initial, cfg):
        best = np.zeros(140)
        dm.write(best)
        history.append(dict(label='final_rejected', command=best.tolist(), **initial))
    return best


PACKAGE = Path(__file__).resolve().parent
DEFAULTS = dict(beam=1, phasemask='H3', gain_e_per_adu=2., background_buffer_px=3., halo_margin_px=1.,
    channel=2, mask_axis='BMY', mask_offset_um=200., settle_s=5.,
    motion_timeout_s=20., capture_timeout_s=60., clear_frames=200, frames=50,
    poke_frames=20, poke_amplitude=.03, registration_tolerance_px=.75,
    max_offset=.07, scan_peak=.035, min_scan_peak=.004, scan_points=7,
    iterations=4, max_flux_drift=.10, max_clear_fit_rmse=.2, objective='halo',
    accept_sigma=2., min_rmse_improvement=.0001, bad_pixels_npy=None,
    command_min=None, command_max=None, mds_port=5555, shm_directory='/dev/shm',
    reference_config='embedded:reference_config.json',
    material_file='embedded:material_table')


def load_settings(path, mode, overrides=None, offline=False):
    if mode not in ('online', 'simulation'):
        raise ValueError('Choose online or simulation entry point')
    supplied = json.loads(Path(path).read_text()) if path else {}
    overrides = {key: value for key, value in (overrides or {}).items() if value is not None}
    demo_defaults = dict(settle_s=.001, clear_frames=40, frames=20, poke_frames=10) if offline else {}
    cfg = DEFAULTS | demo_defaults | supplied | overrides
    unknown = (set(supplied) | set(overrides))-set(DEFAULTS)-{'camera_shm'}
    if unknown:
        raise ValueError(f'Unknown/obsolete settings: {sorted(unknown)}; use v2 settings')
    # Endpoints are chosen by the entry point, not carried over from old settings.
    host = MDS_HOST_ONSKY if mode == 'online' else MDS_HOST_SIMULATOR
    cfg['mode'], cfg['mds'] = mode, f"tcp://{host}:{cfg['mds_port']}"
    for key in ('reference_config', 'material_file', 'bad_pixels_npy'):
        if cfg[key] is not None:
            if key in ('reference_config', 'material_file') and cfg[key] == DEFAULTS[key]:
                continue
            value = Path(cfg[key])
            if not value.is_absolute():
                value = (Path(path).resolve().parent if path else PACKAGE)/value
            cfg[key] = str(value)
    if any(cfg[k] is None for k in ('command_min', 'command_max')):
        raise ValueError('Set command_min and command_max for the summed DM command in settings')
    if cfg['beam'] not in (1, 2, 3, 4) or cfg['phasemask'] not in [f'{b}{i}' for b in 'HJ' for i in range(1, 6)]:
        raise ValueError('Invalid beam or phase mask')
    if cfg['channel'] < 1 or cfg['mask_axis'] not in ('BMX', 'BMY') or cfg['objective'] not in ('rmse', 'halo'):
        raise ValueError('Invalid channel, mask axis or objective')
    positive = ['gain_e_per_adu', 'background_buffer_px', 'settle_s', 'capture_timeout_s',
        'motion_timeout_s', 'poke_amplitude', 'max_offset', 'scan_peak', 'min_scan_peak',
        'registration_tolerance_px', 'max_flux_drift', 'max_clear_fit_rmse', 'accept_sigma']
    for key in positive:
        if not np.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    for key in ('clear_frames', 'frames', 'poke_frames', 'iterations', 'scan_points', 'channel', 'mds_port'):
        if type(cfg[key]) is not int or cfg[key] < (3 if 'frames' in key else 1):
            raise ValueError(f'Invalid integer setting: {key}')
    if cfg['clear_frames'] < 6 or cfg['mds_port'] > 65535:
        raise ValueError('Need >=6 clear frames and a valid MDS port')
    if not 1 <= cfg['iterations'] <= 5:
        raise ValueError('iterations must be between 1 and 5')
    if cfg['scan_points'] not in (3, 5, 7, 9):
        raise ValueError('scan_points must be one of 3, 5, 7, 9')
    if not np.isfinite(cfg['halo_margin_px']) or cfg['halo_margin_px'] < 0:
        raise ValueError('halo_margin_px must be finite and nonnegative')
    if not np.isfinite(cfg['min_rmse_improvement']) or cfg['min_rmse_improvement'] < 0:
        raise ValueError('min_rmse_improvement must be nonnegative')
    if not np.isfinite([cfg['command_min'], cfg['command_max'], cfg['mask_offset_um']]).all() or cfg['command_min'] >= cfg['command_max'] or cfg['mask_offset_um'] == 0:
        raise ValueError('Invalid hardware bounds or mask displacement')
    if cfg['poke_amplitude'] > cfg['max_offset'] or cfg['scan_peak'] > cfg['max_offset']:
        raise ValueError('Poke/scan amplitude exceeds cumulative bound')
    return cfg


def run(cfg, output, apply_best=False, plot=False, backend=None, confirm=input, optimizer=None):
    cfg = dict(cfg)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output/'settings.json').write_text(json.dumps(cfg, indent=2))
    model_config = load_config(cfg['reference_config'])
    model_config['phase_mask']['name'] = cfg['phasemask']
    background_adu = None
    good = None
    arrays, history = {}, []
    camera = device = dm = None
    leave_best = False
    state = dict(status='running', dm_state='unchanged')
    try:
        if backend is None:
            camera = Camera(cfg.get('camera_shm', str(Path(cfg['shm_directory'])/f"baldr{cfg['beam']}.im.shm")))
            device = Device(beam_id=cfg['beam'], main_chn=cfg['channel'], shm_directory=cfg['shm_directory'])
            call = lambda command: rpc(cfg['mds'], command)
        else:
            camera, device, call = backend.camera, backend.device, backend.call
        dm = DM(device, cfg['channel'], (cfg['command_min'], cfg['command_max']), cfg['max_offset'])
        arrays['initial_dm_channels'] = dm.initial
        np.save(output/'starting_dm_channels.npy', dm.initial)
        def capture(label, count):
            dm.check_owner()
            time.sleep(cfg['settle_s'])
            raw, counters = fresh_frames(camera, count, cfg['capture_timeout_s'])
            dm.check_owner()
            arrays[label+'_raw_adu'] = raw
            arrays[label+'_counters'] = counters
            if background_adu is None:
                return raw
            return reduce_frames(raw, background_adu, cfg['gain_e_per_adu'], good)
        axis = f"{cfg['mask_axis']}{cfg['beam']}"
        print(f"Starting from current DM state and current {cfg['phasemask']} mask position.")
        with mask_out(call, axis, cfg['mask_offset_um'], cfg['settle_s'], cfg['motion_timeout_s']) as original:
            state['mask_axis'], state['mask_original'] = axis, original
            (output/'state.json').write_text(json.dumps(state, indent=2))
            raw_clear = capture('clear', cfg['clear_frames'])
        bad = np.load(cfg['bad_pixels_npy'], allow_pickle=False) if cfg['bad_pixels_npy'] else None
        detector, good, background_mask = estimate_detector(raw_clear, cfg['gain_e_per_adu'], cfg['background_buffer_px'], bad)
        background_adu = detector['background_adu']  # Frozen for this entire run.
        cfg['_effective_noise_e'] = detector['effective_noise_e']
        state['detector'] = detector
        print(f"Fixed background={background_adu:.3f} ADU; effective noise={detector['effective_noise_adu']:.3f} ADU")
        clear_cube = reduce_frames(raw_clear, background_adu, cfg['gain_e_per_adu'], good)
        arrays.update(background_adu=np.array(background_adu), background_mask=background_mask,
                      good=good, effective_noise_e=np.array(cfg['_effective_noise_e']))
        while True:
            answer = confirm('Mask restored. Check/adjust alignment externally; enter y to capture ZWFS, q to abort: ').strip().lower()
            if answer == 'y':
                break
            if answer in ('q', 'n'):
                raise RuntimeError('Operator aborted alignment')
        initial_cube = capture('initial_zwfs', cfg['frames'])
        clear = clear_cube.mean(axis=0)
        # Fill excluded pixels ONLY for geometry; they remain excluded from metrics.
        geometry_image = fill_for_geometry(clear, good)
        geometry = detector['geometry']
        cx, cy, radius = (float(geometry[k]) for k in ('center_x', 'center_y', 'radius_pixels'))
        halo_radius = radius+cfg['halo_margin_px']
        pupil = halo_pupil(clear.shape, cx, cy, radius, cfg['halo_margin_px'])
        print(f"Clear radius={radius:.3f} px; halo boundary={halo_radius:.3f} px (margin {cfg['halo_margin_px']:g})")
        yy, xx = np.indices(clear.shape)
        # Keep normalization/registration support independent of halo tuning.
        normalization_pupil = np.hypot(xx-cx, yy-cy) <= radius+.25
        smooth = ndimage.gaussian_filter(np.maximum(geometry_image, 0), .7)
        support = ndimage.binary_erosion(smooth > .08*smooth.max()) & normalization_pupil & good
        if support.sum() < 12 or clear[support].mean() <= 0:
            raise RuntimeError('Insufficient clear-pupil normalization support')
        model_config.setdefault('detector', {})['crop'] = list(clear.shape)
        (output/'model_config.json').write_text(json.dumps(model_config, indent=2))
        theory_clear, theory_masked, wavelengths = generate_references(model_config, Path(cfg['material_file']))
        if theory_clear.shape != clear.shape or not np.isfinite(theory_clear).all() or not np.isfinite(theory_masked).all():
            raise ValueError('Invalid theoretical reference images')
        tc, target_image, params, fit_error = fit_theory(geometry_image, theory_clear,
            theory_masked, support, good, estimate_pupil_center_subpixel, cfg['max_clear_fit_rmse'])
        matrix, active, xy, pokes, errors, heldout = register_dm(dm, capture, support, good, cfg, registration)
        target = matrix @ target_image.ravel()
        metrics = Metrics(clear_cube, pupil, support, good, matrix, active, target, cfg)
        metrics.measure(initial_cube)
        arrays.update(clear_e=clear, pupil=pupil, support=support, good=good, I2A=matrix,
            active_actuators=active, actuator_xy=xy, theory_clear=tc, theory_zwfs=target_image,
            theory_dm=target, registration_pokes=pokes, transform=params, wavelengths=wavelengths,
            clear_raw_mean=raw_clear.mean(axis=0))
        state.update(pupil=dict(x=cx, y=cy, radius=radius, halo_radius=halo_radius, halo_margin_px=cfg['halo_margin_px']),
            clear_fit_rmse=fit_error, registration_errors=errors.tolist(), heldout_error=heldout,
            initial_halo_excess=metrics.initial_excess)
        best = (optimize if optimizer is None else optimizer)(dm, capture, metrics, cfg, history)
        arrays['best_increment_140'] = best
        arrays['best_channel_map'] = dm.initial[cfg['channel']]+dm_map(best)
        arrays['best_total_map'] = dm.initial.sum(axis=0)+dm_map(best)
        np.savetxt(output/'incremental_offset_140.txt', best, fmt='%.9g')
        np.savez_compressed(output/'calibration.npz', **arrays)
        if plot:
            summary_plot(output, arrays, history)
        # All requested result writes must succeed before committing a live result.
        if optimizer is not None and hasattr(optimizer, 'diagnostics'):
            (output/'nonlinear_diagnostics.json').write_text(json.dumps(optimizer.diagnostics, indent=2))
        (output/'history.json').write_text(json.dumps(history, indent=2))
        state['status'] = 'complete'
        (output/'state.json').write_text(json.dumps(state, indent=2))
        leave_best = apply_best
    except BaseException as exc:
        state.update(status='failed', error=str(exc))
        raise
    finally:
        try:
            if dm is not None:
                if leave_best:
                    try:
                        dm.check_owner()
                    except BaseException:
                        dm.restore()
                        raise
                    state['dm_state'] = 'best increment left on selected channel'
                else:
                    dm.restore()
                    state['dm_state'] = 'starting selected channel restored'
        except BaseException as exc:
            state.update(status='failed', dm_state='RESTORATION/OWNERSHIP UNVERIFIED', recovery_error=str(exc))
            raise
        finally:
            try:
                try:
                    if camera is not None:
                        camera.close()
                finally:
                    if device is not None:
                        device.close(erase_file=False)
            finally:
                # Retain failed scans too, but only after attempting hardware recovery.
                if optimizer is not None and hasattr(optimizer, 'diagnostics'):
                    (output/'nonlinear_diagnostics.json').write_text(json.dumps(optimizer.diagnostics, indent=2))
                np.savez_compressed(output/'calibration.npz', **arrays)
                (output/'history.json').write_text(json.dumps(history, indent=2))
                (output/'state.json').write_text(json.dumps(state, indent=2))
    return output


def summary_plot(output, arrays, history):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].imshow(arrays['clear_e'], origin='upper')
    xy = arrays['actuator_xy'][arrays['active_actuators']]
    axes[0].scatter(xy[:, 0], xy[:, 1], s=8, c='r')
    axes[0].contour(arrays['pupil'].astype(float), levels=[.5], colors=['cyan'], linewidths=1)
    axes[0].set_title('Clear / DM samples / cyan halo boundary')
    axes[1].plot([row['rmse'] for row in history], '.-')
    axes[1].set_title('Pupil RMSE'); axes[1].set_xlabel('Measurement')
    axes[2].plot([row['relative_halo'] for row in history], '.-')
    axes[2].set_title('Relative halo proxy'); axes[2].set_xlabel('Measurement')
    fig.tight_layout(); fig.savefig(output/'summary.png', dpi=150); plt.close(fig)


# None preserves values from JSON; only explicitly supplied flags override them.
CLI_SETTINGS = ('beam', 'phasemask', 'objective', 'iterations', 'scan_points', 'frames',
                'clear_frames', 'poke_frames', 'settle_s', 'scan_peak',
                'background_buffer_px', 'halo_margin_px', 'gain_e_per_adu')


def build_parser(mode):
    parser = argparse.ArgumentParser(description=f'Baldr flat calibration: {mode}',
        epilog='Settings precedence: command-line overrides > JSON > built-in defaults. '
               'The requested phase mask must already be physically selected.')
    parser.add_argument('--settings', type=Path, default=PACKAGE/f'settings.{mode}.json')
    parser.add_argument('--output', type=Path, default=Path('runs')/f"{mode}_{time.strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument('--apply-best', action='store_true', help='Leave verified increment on selected DM channel')
    parser.add_argument('--plot', action='store_true', help='Save one summary figure (requires Matplotlib)')
    parser.add_argument('--beam', type=int, choices=range(1, 5), help='Beam 1–4 (default from JSON, otherwise 1)')
    parser.add_argument('--phasemask', type=str.upper, choices=[f'{band}{i}' for band in 'HJ' for i in range(1, 6)],
                        help='Reference-model mask; select/align the physical mask externally')
    parser.add_argument('--iterations', '--trials', dest='iterations', type=int, choices=range(1, 6),
                        help='Maximum full optimization rounds, 1–5 (default 4); each contains a scan')
    parser.add_argument('--scan-points', type=int, choices=(3, 5, 7, 9),
                        help='Points per round INCLUDING the incumbent; default 7 = 6 nonzero trials')
    parser.add_argument('--frames', type=int, help='Fresh frames per metric window (>=3; default 50)')
    parser.add_argument('--clear-frames', type=int, help='Clear sequence size (>=6; default 200)')
    parser.add_argument('--poke-frames', type=int, help='Frames at each registration poke sign (>=3; default 20)')
    parser.add_argument('--settle-s', type=float, help='Positive settling delay before capture, seconds (default 5)')
    parser.add_argument('--scan-peak', type=float, help='Initial peak scan increment (default 0.035; bounded by max_offset)')
    parser.add_argument('--background-buffer-px', type=float, help='Exterior buffer for the one background estimate (default 3 pixels)')
    parser.add_argument('--halo-margin-px', type=float, help='Halo inside radius = fitted clear radius + margin (default 1 pixel; >=0)')
    parser.add_argument('--objective', type=str.lower, choices=('halo', 'rmse'),
                        help='Primary optimization metric (default halo); reject significant degradation in the other metric')
    parser.add_argument('--gain', dest='gain_e_per_adu', type=float, help='Approximate electrons/ADU (default 2)')
    if mode == 'simulation':
        parser.add_argument('--offline', action='store_true', help='Illustrative fake instrument; no SHM/network; smaller acquisition defaults')
    return parser


def main(mode, configure_parser=None, optimizer_factory=None):
    parser = build_parser(mode)
    if configure_parser is not None:
        configure_parser(parser)
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    offline = getattr(args, 'offline', False)
    overrides = {key: getattr(args, key) for key in CLI_SETTINGS}
    try:
        cfg = load_settings(args.settings, mode, overrides=overrides, offline=offline)
        optimizer = None if optimizer_factory is None else optimizer_factory(args, cfg)
    except ValueError as exc:
        parser.error(str(exc))
    backend = None
    confirm = input
    if offline:
        cfg['mode'] = 'offline_demo'
        backend = OfflineInstrument(cfg)
        confirm = lambda prompt: 'y'
        print('OFFLINE DEMO: synthetic detector response; no optical performance claim.')
    else:
        print(f"{mode.upper()}: MDS={cfg['mds']}, SHM={cfg['shm_directory']}, beam={cfg['beam']}, mask={cfg['phasemask']}")
    print(f"Objective: {cfg['objective']} (the other metric remains a degradation guard)")
    print(f"At most {cfg['iterations']} rounds, {cfg['scan_points']-1} nonzero trials per round; "
          f"{cfg['frames']} frames/window, {cfg['clear_frames']} clear frames, settle={cfg['settle_s']:g} s")
    result = run(cfg, args.output, args.apply_best, args.plot, backend, confirm, optimizer=optimizer)
    print(f'Calibration saved: {result.resolve()}')


# Offline

from pathlib import Path
import numpy as np


class Stream:
    def __init__(self, value):
        self.value = np.array(value, dtype=np.float32)
    def get_data(self):
        return self.value.copy()
    def set_data(self, value):
        self.value = np.asarray(value, dtype=np.float32).copy()
    def post_sems(self, number):
        pass
    def close(self, erase_file=False):
        pass


class FakeDevice:
    def __init__(self):
        self.shms = [Stream(np.full((12,12),.5)), Stream(np.zeros((12,12))), Stream(np.zeros((12,12)))]
        self.shm0 = Stream(np.full((12,12),.5))
        self.closed = False
    def close(self, erase_file=False):
        self.closed = True


class OfflineInstrument:
    def __init__(self, cfg):
        self.cfg, self.device, self.camera = cfg, FakeDevice(), self
        self.stage, self.counter = 0., 0
        self.rng = np.random.default_rng(418)
        config = load_config(cfg['reference_config'])
        config['phase_mask']['name'] = cfg['phasemask']
        config['detector']['crop'] = [32,32]
        clear, masked, _ = generate_references(config, Path(cfg['material_file']))
        scale = 20000./clear.max()
        self.clear, self.masked = clear*scale, masked*scale  # ADU before gain.
        geometry = estimate_pupil_center_subpixel(clear)
        y,x = np.indices(clear.shape)
        distance = np.hypot(x-geometry['center_x'],y-geometry['center_y'])
        pupil = distance <= geometry['radius_pixels']
        coords = np.asarray(generate_dm_coordinates()[0])
        xy = coords*(geometry['radius_pixels']/5.3)+[geometry['center_x'],geometry['center_y']]
        self.influence = np.array([np.exp(-((x-px)**2+(y-py)**2)/(.8**2*2))*pupil for px,py in xy])
        self.aberration = .012*np.sin(coords[:,0]/2)*np.cos(coords[:,1]/3)
        self.active = np.ones((12,12),bool)
        self.active[0,0]=self.active[0,-1]=self.active[-1,0]=self.active[-1,-1]=False
        self.halo_shape = np.where(distance>geometry['radius_pixels']+.25,masked,0.)
        self.inner_shape = np.where(pupil,clear,0.)
        self.inner_shape /= self.inner_shape.sum()
        self.background_adu = 850.
        self.noise_adu = 4.
        self.closed = False

    def call(self, command):
        if command.startswith('read '):
            return str(self.stage)
        if command.startswith('moverel '):
            self.stage += float(command.split()[-1])
            return 'ACK'
        raise ValueError(command)

    def read(self):
        self.counter += 1
        offset = self.device.shms[self.cfg['channel']].get_data()[self.active]
        error = offset+self.aberration
        if abs(self.stage) > 100:
            signal = self.clear.copy()
        else:
            signal = self.masked.copy()
            signal += 8*self.clear.max()*np.tensordot(error,self.influence,axes=1)
            # Degrade the halo at nonzero command error, returning flux inside.
            penalty = min(.4, .12*np.mean(error**2)/np.mean(self.aberration**2))
            lost = penalty*self.halo_shape
            signal += -lost+lost.sum()*self.inner_shape
        signal = np.maximum(signal,0.)
        gain = self.cfg['gain_e_per_adu']
        frame = self.background_adu+self.rng.poisson(signal*gain)/gain
        frame += self.rng.normal(0,self.noise_adu,signal.shape)
        return self.counter, frame

    def close(self):
        self.closed = True


if __name__ == '__main__':
    main('simulation')
