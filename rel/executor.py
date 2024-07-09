# 3) work on rel executor with REL sdk: 2 hours, WIP
import functools
import sys
import threading
import time
import json
import traceback
from contextlib import nullcontext
from urllib.error import HTTPError
from os import path
from urllib.request import urlopen

from railib import api, config, show
from railib.credentials import ClientCredentials
from railib.rest import CLIENT_ID_KEY, _default_user_agent, _get_host, CLIENT_SECRET_KEY, AUDIENCE_KEY, GRANT_TYPE_KEY, \
    CLIENT_CREDENTIALS_KEY, _encode, Request, _print_request, ACCESS_KEY_TOKEN_KEY, EXPIRES_IN_KEY, SCOPE, \
    AccessToken

from toolkit.util import project_root


def table_id_to_rai_relation_name(table_id, prefix="csv"):
    csv_name = f"{prefix}_" + table_id.replace("-", "_")
    return csv_name


def is_term_state(state: str) -> bool:
    return state == "DELETED" or ("FAILED" in state)


def show_error(fun):
    @functools.wraps(fun)
    def wrapped(*args, **kwargs):
        try:
            return fun(*args, **kwargs)
        except HTTPError as e:
            show.http_error(e)

    return wrapped


def _sansext(fname: str) -> str:
    return path.splitext(path.basename(fname))[0]


PROFILE = "default"


class AccessTokenManager:

    def __init__(self):
        self.lock = threading.Lock()
        self.access_token = None
        self.access_token_last_requested = None
        con = config.read(profile=PROFILE)
        host = con["host"]
        self.url = f"https://{host}:443/transactions"

    def refresh(self, ctx):
        # when entering the execution, manually check and refresh the token if needed
        url = self.url  # 'https://azure.relationalai.com:443/transactions'
        with self.lock:
            if self.access_token is None or self.token_expired():
                self.kill_if_access_frequently()
                self.access_token = self.request_access_token(ctx, url)
                if self.access_token is None:
                    sys.exit(1)
                if self.token_expired():
                    sys.exit(1)
                print("requested access token")
            ctx.credentials.access_token = self.access_token

    def kill_if_access_frequently(self):
        if self.access_token_last_requested is not None:
            if time.time() - self.access_token_last_requested < 2000:
                sys.exit(1)
        self.access_token_last_requested = time.time()

    def token_expired(self):
        t = self.access_token.expires_in - time.time() + self.access_token.created_on
        return t < 1200

    def request_access_token(self, ctx, url):
        creds = ctx.credentials
        assert isinstance(creds, ClientCredentials)
        # ensure the audience contains the protocol scheme
        audience = ctx.audience or f"https://{_get_host(url)}"
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Host": _get_host(creds.client_credentials_url),
            "User-Agent": _default_user_agent(),
        }
        body = {
            CLIENT_ID_KEY: creds.client_id,
            CLIENT_SECRET_KEY: creds.client_secret,
            AUDIENCE_KEY: audience,
            GRANT_TYPE_KEY: CLIENT_CREDENTIALS_KEY,
        }
        data = _encode(body)
        req = Request(
            method="POST",
            url=creds.client_credentials_url,
            headers=headers,
            data=data,
        )
        _print_request(req)
        with urlopen(req) as rsp:
            result = json.loads(rsp.read())
            token = result.get(ACCESS_KEY_TOKEN_KEY, None)

            if token is not None:
                expires_in = result.get(EXPIRES_IN_KEY, None)
                scope = result.get(SCOPE, None)
                return AccessToken(token, scope, expires_in)

        raise Exception("failed to get the access token")


# access_token_manager = AccessTokenManager()


class RELX:
    def __init__(self, engine_size="S", postfix="", verbose=False):
        self.engine_name = f"neural_interpretation_dev{postfix}"
        self.database_name = f"neural_interpretation_dev_rerun_db{postfix}"
        self.cfg, self.ctx = None, None
        self.profile = PROFILE  # "default"
        self.state = None
        self.engine_size = engine_size
        self.verbose = verbose
        # self.access_token_manager = access_token_manager
        self.setup()

    @show_error
    def setup(self):
        self.cfg = config.read(profile=self.profile)
        # self.cfg["retries"] = 3
        self.ctx = api.Context(**self.cfg)

    @show_error
    def list_all_engines(self):
        # self.access_token_manager.refresh(self.ctx)
        rsp = api.list_engines(self.ctx, state=self.state)
        return rsp

    @show_error
    def list_all_databases(self):
        # self.access_token_manager.refresh(self.ctx)
        rsp = api.list_databases(self.ctx, state=self.state)
        return rsp

    def engine_is_created(self):
        # self.access_token_manager.refresh(self.ctx)
        rsp = self.list_all_engines()
        for engine in rsp:
            if engine['name'] == self.engine_name:
                return True
        return False

    def database_is_created(self):
        rsp = self.list_all_databases()
        for db in rsp:
            if db['name'] == self.database_name:
                return True
        return False

    def make_engine(self):
        if self.engine_is_created():
            print("Engine already exists. Skipping creation.")
        else:
            # self.access_token_manager.refresh(self.ctx)
            api.create_engine_wait(self.ctx, self.engine_name, self.engine_size)
            ret = api.get_engine(self.ctx, self.engine_name)
            print(f"Making engine {self.engine_name}")
            return ret

    def make_database(self):
        if self.database_is_created():
            print("Database already exists. Skipping creation.")
        else:
            # self.access_token_manager.refresh(self.ctx)
            rsp = api.create_database(self.ctx, self.database_name)
            return rsp

    def shutdown_engine(self):
        while True:  # wait for request to reach terminal state
            time.sleep(3)
            # self.access_token_manager.refresh(self.ctx)
            rsp = api.delete_engine(self.ctx, self.engine_name)
            rsp = api.get_engine(self.ctx, self.engine_name)
            if not rsp or is_term_state(rsp["state"]):
                break
        return rsp

    # @show_error
    # def upload_csv_file(self, csv_path, relation_name):
    #     with open(csv_path, 'rt') as f:
    #         data = f.read()
    #     syntax = {
    #         "header_row": 1,
    #     }
    #     cfg = config.read(profile=self.profile)
    #     ctx = api.Context(**cfg)
    #     rsp = api.load_csv(ctx, self.database_name, self.engine_name, relation_name, data, syntax)
    #     if self.verbose:
    #         print(json.dumps(rsp, indent=2))
    #
    # @show_error
    # def upload_csv_string(self, csv_string, relation_name):
    #     data = csv_string
    #     syntax = {
    #         "header_row": 1,
    #     }
    #     cfg = config.read(profile=self.profile)
    #     ctx = api.Context(**cfg)
    #     rsp = api.load_csv(ctx, self.database_name, self.engine_name, relation_name, data, syntax)
    #     if self.verbose:
    #         print(json.dumps(rsp, indent=2))

    @show_error
    def upload_json_file(self, json_path, relation_name):
        # self.access_token_manager.refresh(self.ctx)
        self.delete_relation(relation_name)
        with open(json_path, 'rt') as f:
            data = f.read()
        rsp = api.load_json(self.ctx, self.database_name, self.engine_name, relation_name, data)
        if self.verbose:
            print(json.dumps(rsp, indent=2))

    @show_error
    def upload_csv_no_schema(self, csv_string, relation_name):
        # self.access_token_manager.refresh(self.ctx)
        syntax = {
            "header_row": 1,
            "escapechar": '"'
        }
        self.delete_relation(relation_name)
        rsp = api.load_csv(self.ctx, self.database_name, self.engine_name, relation_name, csv_string, syntax)
        print(json.dumps(rsp, indent=2))

    @show_error
    def upload_csv(self, csv_string, relation_name, schema=None):
        # self.access_token_manager.refresh(self.ctx)
        syntax = {
            "header_row": 1,
            "escapechar": '\'"\''
        }
        self.delete_relation(relation_name)
        # https://docs.relational.ai/rkgms/sdk/python-sdk#querying-a-database
        query = ""
        for k, v in schema.items():
            query += f'def config:schema:{k}="{v}"\n'

        for k, v in syntax.items():
            query += f'def config:syntax:{k}={v}\n'
        query += f"def config:data = mydata\n"
        query += f"def insert[:{relation_name}] = load_csv[config]\n"

        rsp = api.exec(self.ctx, self.database_name, self.engine_name, query, {"mydata": csv_string}, readonly=False)

    @show_error
    def upload_json(self, json_string, relation_name):
        # self.access_token_manager.refresh(self.ctx)
        self.delete_relation(relation_name)
        rsp = api.load_json(self.ctx, self.database_name, self.engine_name, relation_name, json_string)
        print(json.dumps(rsp, indent=2))

    def delete_relation(self, relation_name):
        query = f"def delete[:{relation_name}] = {relation_name}"
        self.execute_query(query, print_error=False)

    def execute_query(self, query, print_error=True, timeout=1000):
        """
        Query is stateless: it is not an interpreter and does not memorize previously executed queries.

        :param query:
        :return:
        """
        # self.access_token_manager.refresh(self.ctx)
        # retry = 3
        # while retry > 0:
        try:
            rsp = api.exec(self.ctx, self.database_name, self.engine_name, query, timeout=timeout, readonly=False)
            if len(rsp.problems) > 0 and print_error:
                show.results(rsp)
            return rsp
        except HTTPError:
            traceback.print_exc()
            return None
        except Exception as e:
            traceback.print_exc()
            if e.args[0] == f"timed out after {timeout} seconds":
                raise TimeoutError
            else:
                return None

    def execute_rel_query_against_table(self, query, table_id, print_error=False):
        csv_id = table_id_to_rai_relation_name(table_id, 'csv')
        code = f"""
        def tbl = {csv_id}
        {query}
        """
        rsp = self.execute_query(code, print_error=print_error)
        ret = self.rsp_to_single_result(rsp)
        return ret

    def rsp_to_single_result(self, rsp):
        for i, res in enumerate(rsp.results):
            for v in zip(*res["table"].to_pydict().values()):
                return v[0]

    def to_rel_schema(self, wikisql_schema):
        map = {
            "text": "string",
            "real": "float",

        }
        ret = {}
        for k, v in wikisql_schema.items():
            ret[k] = map[v]
        return ret

    def main_create_engine_database(self):
        # r = self.list_all_engines()
        # print(r)
        # r = self.engine_is_created()
        # print(r)
        r = self.make_engine()
        # r = self.database_is_created()
        # print(r)
        r = self.make_database()


program1 = """
module config
    def path = "azure://raidocs.blob.core.windows.net/csv-import/simple-import-4cols.csv"
 
    def schema = {
        (:cocktail, "string");
        (:quantity, "int");
        (:price, "decimal(64, 2)");
        (:date, "date");
    }
end

// import csv data as a base relation
def insert:my_data = load_csv[config]
def output(c,d) = my_data:cocktail(pos, c) and my_data:date(pos, d) from pos where c = "martini"
"""

if __name__ == '__main__':
    # for testing purposes
    relx = RELX()
    relx.main_create_engine_database()

    f_path = project_root() / "data/wikisql/dev/tables/2-17360840-6.csv"
    relation_name = "csv1"
    relx.upload_csv_file(f_path, relation_name)
    print("uploaded")
    rsp = relx.execute_query(program1)
    print(rsp)
    # rsp = relx.execute_query("def output = 2")
    # print(rsp)

    # relx.execute_query(
    #     "def output(c,d) = my_data:cocktail(pos, c) and my_data:date(pos, d) from pos where c = 'martini'")

    # relx.shutdown_engine()
