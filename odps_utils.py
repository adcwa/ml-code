# -*- coding: utf-8 -*-
import os
import json
import logging
from tqdm import tqdm
from odps import ODPS
from odps.accounts import AliyunAccount
from odps.accounts import StsAccount


ODPS_CREDENTIALS_FILE: str = "/ml/input/credential/odps.json"
ODPS_META_FILE: str = "meta.json"
INSTRUCTION_KEY = 'instruction'
OUTPUT_KEY = 'output'


def is_odps_table(data_dir: str) -> bool:
    return (
        os.path.exists(data_dir)
        and os.path.isdir(data_dir)
        and os.path.exists(ODPS_CREDENTIALS_FILE)
        and os.path.exists(os.path.join(data_dir, ODPS_META_FILE))
    )


def parse_input_table_path(odps_table_path):
    # input table path format: odps://project_name/tables/table_name
    str_list = odps_table_path.split("/")
    if len(str_list) < 5 or str_list[3] != "tables":
        raise ValueError(
            "'%s' is invalid, please refer: 'odps://${your_projectname}/"
            "tables/${table_name}/${pt_1}/${pt_2}/...'" % odps_table_path
        )

    project_name = str_list[2]
    table_name = str_list[4]

    table_partition = ",".join(str_list[5:])
    if not table_partition:
        table_partition = None

    table_info = {
        "odps project": project_name,
        "table name": table_name,
        "table partition": table_partition,
    }
    logging.info("%s -> %s", odps_table_path, str(table_info))
    return project_name, table_name, table_partition


def build_odps_instance(credentials, end_point, project) -> ODPS:
    access_key_id = credentials["AccessKeyId"]
    access_key_secret = credentials["AccessKeySecret"]
    security_token = credentials["SecurityToken"]

    if not access_key_id or not access_key_secret:
        raise RuntimeError(f"access_key_id or access_key_secret is empty")

    if security_token:
        return ODPS._from_account(
            account=StsAccount(access_key_id, access_key_secret, security_token),
            project=project,
            endpoint=end_point,
        )
    else:
        return ODPS._from_account(
            account=AliyunAccount(access_key_id, access_key_secret),
            project=project,
            endpoint=end_point,
        )


class ODPSTableDataset(object):
    def __init__(self, output_path, instruction_key=INSTRUCTION_KEY, output_key=OUTPUT_KEY):
        self.output_path = output_path
        self.instruction_key = instruction_key
        self.output_key = output_key

    def get_data(self):
        with open(ODPS_CREDENTIALS_FILE) as c:
            credentials = json.load(c)

        data_dir = self.output_path
        with open(os.path.join(data_dir, ODPS_META_FILE)) as m:
            meta = json.load(m)

        end_point = meta["endpoint"]
        project, table_name, _ = parse_input_table_path(meta["path"])
        partitions_str = meta.get("partitions", None)
        partitions = None
        if partitions_str:
            partitions = partitions_str.split(",")

        o = build_odps_instance(credentials, end_point, project)
        t = o.get_table(table_name)

        total_samples = []
        instruction_field = self.instruction_key
        output_field = self.output_key

        if partitions is None:
            with t.open_reader() as reader:
                with tqdm(total=reader.count) as pbar:
                    for record in reader:
                        total_samples.append(
                            {
                                instruction_field: record[instruction_field],
                                output_field: record[output_field]
                            })
                        pbar.update(1)
        else:
            for partition in partitions:
                assert t.exist_partition(partition)
                with t.open_reader(partition=partition) as reader:
                    with tqdm(total=reader.count,  desc=f'{partition}:') as pbar:
                        for record in reader:
                            total_samples.append(
                                {
                                    instruction_field: record[instruction_field],
                                    output_field: record[output_field]
                                })
                            pbar.update(1)
        return total_samples
