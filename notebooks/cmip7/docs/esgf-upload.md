# Validation and publishing to ESGF

Operational steps for getting a finished gridding version out of this repository and onto ESGF.
None of this is automated; none of it is inferable from the code.

## Validation: data format checking for input4MIPs

No validation script exists in this repository yet. Until one does, the relevant references are:

- Writing valid netCDF files:
  <https://input4mips-validation.readthedocs.io/en/latest/how-to-guides/how-to-write-a-single-valid-file/>
- Validating a single file:
  <https://input4mips-validation.readthedocs.io/en/latest/how-to-guides/how-to-validate-a-single-file/>

Writing a validation step against these is open work — see the repository issues.

## Uploading

### Via filesender (the route used so far)

Used for version `0-3-0`:

1. **ZIP the files.** Uncompressed netCDF output is very large; compress before transferring.
2. **Upload** through <https://filesender.aco.net/> (web interface is fine).
3. **Share the download link** with the Forcings TT chairs — in particular Paul Durack and
   Zebedee Nicholls.

### Directly via FTP

Uploading ourselves over FTP is possible but has not been the working route. Reference docs:

- How uploading works:
  <https://input4mips-validation.readthedocs.io/en/latest/how-to-guides/how-to-upload-to-ftp/>
- Where to upload:
  <https://github.com/PCMDI/input4MIPs_CVs/blob/4b59bf4694ddb8bf20f265adcf926046486069bd/docs/usage-data-producer.md#get-your-data-to-pcmdi>

## After publishing

Record the resulting DOI and the per-version changes in the Zenodo documentation record rather
than in this repository — see [`versions.md`](versions.md).
