from soccer_clips.roster import Roster, RosterPlayer, load_roster, roster_path, save_roster


def test_roster_round_trip(settings):
    assert load_roster(settings).numbers() == []
    save_roster(settings, Roster(players=[RosterPlayer(number=38, name="A Player"), RosterPlayer(number=7)]))
    r = load_roster(settings)
    assert r.numbers() == [7, 38] and r.name(38) == "A Player" and r.name(7) is None and r.name(99) is None
    assert roster_path(settings).parent == settings.games_dir  # with the game data, not in the repo
