# SPDX-License-Identifier: Apache-2.0

"""ONNX Model Hub

This implements the python client for the ONNX model hub.
"""
from os.path import join
from urllib.request import urlopen
from urllib.error import HTTPError
import json
import os
import wget  # type: ignore
import hashlib
from io import BytesIO
from typing import List, Optional, Dict, Any, cast, Set, IO
import onnx

if "ONNX_HOME" in os.environ:
    _ONNX_HUB_DIR = join(os.environ["ONNX_HOME"], "hub")
elif "XDG_CACHE_HOME" in os.environ:
    _ONNX_HUB_DIR = join(os.environ["XDG_CACHE_HOME"], "onnx", "hub")
else:
    _ONNX_HUB_DIR = join(os.path.expanduser('~'), ".cache", "onnx", "hub")


class ModelInfo(object):
    def __init__(self, raw_model_info: Dict[str, Any]) -> None:
        self.model = cast(str, raw_model_info["model"])

        self.model_path = cast(str, raw_model_info["model_path"])
        self.metadata: Dict[str, Any] = cast(Dict[str, Any], raw_model_info["metadata"])
        self.model_sha: Optional[str] = None
        if "model_sha" in self.metadata:
            self.model_sha = cast(str, self.metadata["model_sha"])

        self.tags: Set[str] = set()
        if "tags" in self.metadata:
            self.tags = set(cast(List[str], self.metadata["tags"]))

        self.opset = cast(int, raw_model_info["opset_version"])
        self.raw_model_info: Dict[str, Any] = raw_model_info

    def __str__(self) -> str:
        return "ModelInfo(model={}, opset={}, path={}, metadata={})".format(
            self.model, self.opset, self.model_path, self.metadata
        )

    def __repr__(self) -> str:
        return self.__str__()


def set_dir(new_dir: str) -> None:
    """
    Set the current ONNX hub cache location
    @param new_dir: location of new model hub cache
    """
    global _ONNX_HUB_DIR
    _ONNX_HUB_DIR = new_dir


def get_dir() -> str:
    """
    Get the current ONNX hub cache location
    @return: The location of the ONNX hub model cache
    """
    return _ONNX_HUB_DIR


def _get_base_url(repo: str, lfs: bool = False) -> str:
    """
    Gets the base github url from a repo specification string
    @param repo: The location of the model repo in format "user/repo[:branch]".
        If no branch is found will default to "master"
    @param lfs: whether the url is for downloading lfs models
    @return: the base github url for downloading
    """
    repo_owner = repo.split("/")[0]
    repo_name = repo.split("/")[1].split(":")[0]
    if ":" in repo:
        repo_branch = repo.split("/")[1].split(":")[1]
    else:
        repo_branch = "master"

    if lfs:
        return "https://media.githubusercontent.com/media/{}/{}/{}/".format(repo_owner, repo_name, repo_branch)
    else:
        return "https://raw.githubusercontent.com/{}/{}/{}/".format(repo_owner, repo_name, repo_branch)


def list_models(repo: str = "onnx/models:master", tags: Optional[List[str]] = None) -> List[ModelInfo]:
    """
        Get the list of model info consistent with a given name and opset

        @param repo: The location of the model repo in format "user/repo[:branch]".
            If no branch is found will default to "master"
        @param tags: A list of tags to filter models by
        """
    base_url = _get_base_url(repo)
    manifest_url = base_url + "ONNX_HUB_MANIFEST.json"
    try:
        with urlopen(manifest_url) as f:
            manifest: List[ModelInfo] = [ModelInfo(info) for info in json.load(cast(IO[str], f))]
    except HTTPError as e:
        raise AssertionError("Could not find manifest at {}".format(manifest_url), e)

    if tags is None:
        return manifest
    else:
        canonical_tags = {t.lower() for t in tags}
        matching_info_list: List[ModelInfo] = []
        for m in manifest:
            model_tags = {t.lower() for t in m.tags}
            if len(canonical_tags.intersection(model_tags)) > 0:
                matching_info_list.append(m)
        return matching_info_list


def get_model_info(model: str,
                   repo: str = "onnx/models:master",
                   opset: Optional[int] = None) -> List[ModelInfo]:
    """
    Get the list of model info consistent with a given name and opset

    @param model: The name of the onnx model in the manifest. This field is case-sensitive
    @param repo: The location of the model repo in format "user/repo[:branch]".
        If no branch is found will default to "master"
    @param opset: The opset of the model to download. The default of `None`  will return all models of matching name
    """
    manifest = list_models(repo)
    matching_models = [m for m in manifest if m.model == model]
    assert len(matching_models) != 0, "No models found with name {}".format(model)

    if opset is None:
        selected_models = sorted(matching_models, key=lambda m: m.opset)
    else:
        selected_models = [m for m in matching_models if m.opset == opset]
        if len(selected_models) == 0:
            valid_opsets = [m.opset for m in matching_models]
            raise AssertionError(
                "{} has no version with opset {}. Valid opsets: {}".format(model, opset, valid_opsets))
    return selected_models


def load(model: str,
         repo: str = "onnx/models:master",
         opset: Optional[int] = None,
         force_reload: bool = False) -> onnx.ModelProto:
    """
    Download a model by name from the onnx model hub

    @param model: The name of the onnx model in the manifest. This field is case-sensitive
    @param repo: The location of the model repo in format "user/repo[:branch]".
        If no branch is found will default to "master"
    @param opset: The opset of the model to download. The default of `None` automatically chooses the largest opset
    @param force_reload: Whether to force the model to re-download even if its already found in the cache
    """
    selected_model = get_model_info(model, repo, opset)[0]
    local_model_path_arr = selected_model.model_path.split("/")
    if selected_model.model_sha is not None:
        local_model_path_arr[-1] = "{}_{}".format(selected_model.model_sha, local_model_path_arr[-1])
    local_model_path = join(_ONNX_HUB_DIR, os.sep.join(local_model_path_arr))

    if force_reload or not os.path.exists(local_model_path):
        os.makedirs(os.path.dirname(local_model_path), exist_ok=True)
        lfs_url = _get_base_url(repo, True)
        print("Downloading {} to local path {}".format(model, local_model_path))
        wget.download(lfs_url + selected_model.model_path, local_model_path)
    else:
        print("Using cached {} model from {}".format(model, local_model_path))

    with open(local_model_path, "rb") as f:
        model_bytes = f.read()

    if selected_model.model_sha is not None:
        downloaded_sha = hashlib.sha256(model_bytes).hexdigest()
        assert downloaded_sha == selected_model.model_sha, \
            "Downloaded model has SHA256 {} while checksum is {}".format(downloaded_sha, selected_model.model_sha)

    return onnx.load(cast(IO[bytes], BytesIO(model_bytes)))
