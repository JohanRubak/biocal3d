# BioCal3D workflow sources

For colleague setup and direct 3D labeling, use README.txt and the setup/launcher
at the package root, one level above this folder. The included legacy workflow
launchers now use that same dedicated environment.

Study queues and agreement tools remain available. See LABELING_STUDY.md.
Annotator usage is documented in annotator/README.md. Its previous standalone
setup section is superseded by the root setup scripts and README.txt.

Raw preprocessing is optional: install the root requirements_preprocessing.txt
and pass your raw data folder explicitly using --data-root. Prepared data can be
selected in the labeling interface without raw preprocessing dependencies.
