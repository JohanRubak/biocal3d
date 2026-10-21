Manuscript figures using the supplied TRIOS5 examples only

Figure 1: Mounting arrangement. Insert after the mounting paragraph in Section 2.1.
Figure 2: Experimental overview, including ROI extraction, cleaning stages and scan counts. Panel D covers all G1-G6; there is no study-allocation panel.

Use the 600 dpi PNG files for straightforward insertion in Word. PDF and SVG versions keep text, arrows and rules as vector elements, with the actual scan images embedded. SVG labels can be edited in a vector editor; the scan renderings are raster images. The PNG canvas resolution does not add information to the original 1600-pixel source exports. Figure 1 is designed at 3.5 inches wide and Figure 2 at 7.2 inches wide.

Figure_Captions.txt contains copy-ready captions and suggested placement text. Materials and dimensions have not been inferred from the images. No scale bars were invented. Colours were preserved. Only exterior white margins were removed from the full-model views. All ROI stages within each example share the same crop, preserving their relative display size. Full-model and ROI images use different magnifications.

The assets directory contains seven unchanged source PNGs, all from TRIOS5. source_manifest.json records their source archive paths and checksums. crop_record.json and mounting_annotation_record.json describe the displayed crops and leader targets. build_mounting.py and build_overview.py regenerate the figures using Python, numpy, Pillow and matplotlib.
