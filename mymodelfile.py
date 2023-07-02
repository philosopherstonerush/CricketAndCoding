import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder

class MyModel():
    batter_encode = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    bowler_encode = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    batting_team_encode = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    ven_encode = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    model = RandomForestRegressor(n_estimators=100, max_features=None)

    def __init__(self):
        return None
    
    def fit(self, input):
        dataset = input[0]
        venue = input[1]
        venue = venue.iloc[:, [0, 7]]
        df = pd.merge(dataset, venue, how="outer", on="ID")
        df = df.drop(["ballnumber", 'non-striker', 'extra_type', 'batsman_run', 'extras_run', 'non_boundary', 'isWicketDelivery', 'player_out', 'kind', 'fielders_involved'], axis=1)
        df = df[df['overs'] <=6]
        df = df.drop(['overs'], axis=1)

        df_new = df.groupby(['ID', 'innings']).aggregate({'total_run': 'sum'})
        df = df.drop(["total_run"], axis=1)
        df = df.drop_duplicates()
        df = pd.merge(df, df_new, on=['ID', 'innings'], how='left')

        
        batter_encoded = self.batter_encode.fit_transform(df["batter"].values.reshape(-1, 1))
        df["batter"] = batter_encoded
        bowler_encoded = self.bowler_encode.fit_transform(df["bowler"].values.reshape(-1, 1))
        df["bowler"] = bowler_encoded
        batting_team_encoded = self.batting_team_encode.fit_transform(df["BattingTeam"].values.reshape(-1, 1))
        df["BattingTeam"] = batting_team_encoded
        ven_encoded = self.ven_encode.fit_transform(df["Venue"].values.reshape(-1, 1))
        df["Venue"] = ven_encoded

        # innings, batter, bowler, battingTeam, venue
        x = df.iloc[:, [1, 2, 3, 4, 5]].values
        # total_run
        y= df.iloc[:, [6]].values
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)
        x_test = sc.transform(x_test)

        
        self.model.fit(x_train, np.ravel(y_train, order='C'))

    def predict(self, test_file):
        test_file = test_file.drop(["bowling_team"], axis=1)
        test_file = test_file.assign(batsmen=test_file['batsmen'].str.split(', ')).explode('batsmen')
        test_file = test_file.assign(bowlers=test_file['bowlers'].str.split(', ')).explode('bowlers')
        test_file["venue"] = self.ven_encode.transform(test_file["venue"].values.reshape(-1, 1))
        test_file["batting_team"] = self.batting_team_encode.transform(test_file["batting_team"].values.reshape(-1, 1))
        test_file["batsmen"] = self.batter_encode.transform(test_file["batsmen"].values.reshape(-1, 1))
        test_file["bowlers"] = self.bowler_encode.transform(test_file["bowlers"].values.reshape(-1,1))

        # innings, batter, bowler, battingTeam, venue
        first_innings = test_file[test_file["innings"] == 1]
        second_innings = test_file[test_file["innings"] == 2]
        x_input1 = first_innings.iloc[:, [1, 3, 4, 2, 0]].values
        x_input2 = second_innings.iloc[:, [1, 3, 4, 2, 0]].values
        y1 = np.average(self.model.predict(x_input1))
        y2 = np.average(self.model.predict(x_input2))
        return [y1, y2]