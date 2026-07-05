import gi

gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk


class MyApplication(Gtk.Application):
    def __init__(self):
        super().__init__(application_id="com.example.MyGtkApplication")
        GLib.set_application_name("Speech2MD GUI")

    def do_activate(self):
        window = Gtk.ApplicationWindow(application=self, title="Speech2MD")
        window.present()

app = MyApplication()
app.run()
