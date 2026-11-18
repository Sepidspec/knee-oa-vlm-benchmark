# Upload this package to GitHub manually

The intended repository name is `knee-oa-vlm-benchmark`. The citation files
currently use the connected account's username `Sepidspec`. If you use another
account or repository name, update `CITATION.cff` and `docs/CITATION.md` to
match the actual URL before citing it.

## Create the repository

1. Sign in to your own account at [GitHub](https://github.com).
2. Open [New repository](https://github.com/new).
3. Choose your own account as owner.
4. Set the name to `knee-oa-vlm-benchmark`.
5. Suggested description: `Master thesis: knee osteoarthritis KL grading with Qwen2.5-VL and ResNet-18; reproducible code, recorded results and computational cost.`
6. Select **Public** if thesis readers should be able to open it. If you need
   supervisor review before public release, start private and share access
   with the supervisor; private links require permission.
7. Leave **Add a README**, **Add .gitignore** and **Choose a license** unselected:
   this package already includes README, ignore rules and an explicit rights
   notice. You can choose a different code license later if you wish.
8. Click **Create repository**.

## Upload the prepared files

1. Download and extract `Knee_OA_GitHub_Repository.zip` on your computer.
2. Open the extracted `knee-oa-vlm-benchmark` directory. Its contents include
   `README.md`, `CITATION.cff`, `notebooks`, `scripts`, `src`, `docs`,
   `results`, `artifacts`, `.github`, and `.gitignore`.
3. In the new empty repository, choose **uploading an existing file**. If the
   repository already has files, use **Add file → Upload files**.
4. Drag **the contents inside** the extracted directory into the upload area.
   Keep the folders intact. `README.md` must be at the repository root, not
   inside an extra wrapper directory.
5. Include `.github` and `.gitignore`; enable hidden-file visibility in your
   file browser if necessary. The `artifacts` folder contains the 15.43 MiB
   analysis ZIP and should be included.
6. Do not upload the outer repository ZIP as the only file. Do not add the
   separate private full model backup or source X-rays.
7. Commit message: `Add thesis code, recorded results and reproducibility documentation`.
8. Click **Commit changes** and wait until the root file list appears.

GitHub browser upload supports files up to 25 MiB and at most 100 files per
upload. This package is within those limits. If your browser does not accept
folder dragging, upload the root files first and the directories in separate
batches, preserving their paths, or publish the directory using GitHub Desktop.

## Check and cite the repository

1. Confirm the README is displayed and opens the confusion-matrix figure.
2. Open `notebooks/Knee_OA_Study.ipynb`; it should render as a notebook.
3. Open the **Actions** tab and inspect **Verify recorded study**. GitHub
   may ask you to enable Actions before the first run.
4. Create a release using tag `v1.0.0`, title `Thesis study v1.0.0`, and a brief
   description of the completed study. Do not add private checkpoints.
5. Copy the actual repository URL from the address bar. Use that link in
   the thesis's code-availability section, and record the release or commit.
6. Use **Cite this repository** for the citation metadata. See `docs/CITATION.md`.

Official references:

- [Creating a repository](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-new-repository)
- [Uploading files](https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository)
- [Citation files](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files)

A repository URL is ready to cite only after you have created the repository
and uploaded the files. The URL written in the prepared citation template is
an intended destination, not proof that the repository already exists.
